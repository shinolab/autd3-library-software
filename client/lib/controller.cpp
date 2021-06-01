// File: controller.cpp
// Project: lib
// Created Date: 05/11/2020
// Author: Shun Suzuki
// -----
// Last Modified: 01/06/2021
// Modified By: Shun Suzuki (suzuki@hapis.k.u-tokyo.ac.jp)
// -----
// Copyright (c) 2021 Hapis Lab. All rights reserved.
//

#include "controller.hpp"

#include <condition_variable>
#include <vector>

#include "core/ec_config.hpp"
#include "core/logic.hpp"
#include "primitive_gain.hpp"

namespace autd {

namespace {
bool ModSentFinished(const core::ModulationPtr& mod) { return mod == nullptr || mod->sent() == mod->buffer().size(); }
bool SeqSentFinished(const core::SequencePtr& seq) { return seq == nullptr || seq->sent() == seq->control_points().size(); }
}  // namespace

bool Controller::is_open() const { return this->_link != nullptr && this->_link->is_open(); }

core::GeometryPtr Controller::geometry() const noexcept { return this->_geometry; }

bool& Controller::silent_mode() noexcept { return this->_silent_mode; }
bool& Controller::reads_fpga_info() noexcept { return this->_read_fpga_info; }

Result<std::vector<uint8_t>, std::string> Controller::fpga_info() {
  const auto num_devices = this->_geometry->num_devices();
  this->_fpga_infos.resize(num_devices);
  if (auto res = this->_link->read(&_rx_buf[0], num_devices * core::EC_INPUT_FRAME_SIZE); res.is_err()) return Err(res.unwrap_err());
  for (size_t i = 0; i < num_devices; i++) this->_fpga_infos[i] = _rx_buf[2 * i];
  return Ok(_fpga_infos);
}

Error Controller::update_ctrl_flag() { return this->send(nullptr, nullptr, true); }

Error Controller::open(core::LinkPtr link) {
  if (is_open())
    if (auto close_res = this->close(); close_res.is_err()) return close_res;

  this->_tx_buf = std::make_unique<uint8_t[]>(this->_geometry->num_devices() * core::EC_OUTPUT_FRAME_SIZE);
  this->_rx_buf = std::make_unique<uint8_t[]>(this->_geometry->num_devices() * core::EC_INPUT_FRAME_SIZE);

  this->_link = std::move(link);
  // this->_stm = std::make_shared<STMController>(this->_link, this->_geometry, &this->_silent_mode, &this->_read_fpga_info);
  return this->_link->open();
}

Error Controller::synchronize(const core::Configuration config) {
  if (!this->is_open()) return Err(std::string("Link is not opened."));

  this->_config = config;
  uint8_t msg_id = 0;
  core::Logic::pack_header(core::COMMAND::INIT_MOD_CLOCK, this->_silent_mode, this->_seq_mode, this->_read_fpga_info, &_tx_buf[0], &msg_id);
  size_t size = 0;
  const auto num_devices = this->_geometry->num_devices();
  core::Logic::pack_sync_body(config, num_devices, &_tx_buf[0], &size);

  if (auto res = this->_link->send(&_tx_buf[0], size); res.is_err()) return res;
  return wait_msg_processed(msg_id, 5000);
}

Error Controller::clear() const { return send_header(core::COMMAND::CLEAR); }

Error Controller::send_header(const core::COMMAND cmd, const size_t max_trial) const {
  if (!this->is_open()) return Err(std::string("Link is not opened."));

  const auto send_size = sizeof(core::RxGlobalHeader);
  uint8_t msg_id = 0;
  core::Logic::pack_header(cmd, this->_silent_mode, this->_seq_mode, this->_read_fpga_info, &_tx_buf[0], &msg_id);
  if (auto res = _link->send(&_tx_buf[0], send_size); res.is_err()) return res;
  return wait_msg_processed(msg_id, max_trial);
}

Error Controller::wait_msg_processed(const uint8_t msg_id, const size_t max_trial) const {
  const auto num_devices = this->_geometry->num_devices();
  const auto buffer_len = num_devices * core::EC_INPUT_FRAME_SIZE;
  for (size_t i = 0; i < max_trial; i++) {
    auto res = this->_link->read(&_rx_buf[0], buffer_len);
    if (res.is_err()) return res;
    if (!res.unwrap()) continue;
    if (core::Logic::is_msg_processed(num_devices, msg_id, &_rx_buf[0])) return Ok(true);

    auto wait = static_cast<size_t>(std::ceil(core::EC_TRAFFIC_DELAY * 1000.0 / core::EC_DEVICE_PER_FRAME * static_cast<double>(num_devices)));
    std::this_thread::sleep_for(std::chrono::milliseconds(wait));
  }

  return Ok(false);
}

Error Controller::close() {
  if (auto res = this->_stm->finish(); res.is_err()) return res;
  if (auto res = this->stop(); res.is_err()) return res;
  if (auto res = this->clear(); res.is_err()) return res;

  auto res = this->_link->close();
  this->_link = nullptr;
  this->_tx_buf = nullptr;
  this->_rx_buf = nullptr;

  return res;
}

Error Controller::stop() { return this->send(gain::NullGain::create(), nullptr, false); }

Error Controller::send(const core::GainPtr& gain, const bool wait_for_sent) { return this->send(gain, nullptr, wait_for_sent); }

Error Controller::send(const core::ModulationPtr& mod) { return this->send(nullptr, mod, true); }

Error Controller::send(const core::GainPtr& gain, const core::ModulationPtr& mod, const bool wait_for_sent) {
  if (!this->is_open()) return Err(std::string("Link is not opened."));

  if (auto res = this->_stm->stop(); res.is_err()) return res;

  if (mod != nullptr)
    if (auto res = mod->build(this->_config); res.is_err()) return res;

  if (gain != nullptr) {
    this->_seq_mode = false;
    if (auto res = gain->build(this->_geometry); res.is_err()) return res;
  }

  size_t size = 0;
  core::Logic::pack_body(gain, &this->_tx_buf[0], &size);

  while (true) {
    uint8_t msg_id = 0;
    core::Logic::pack_header(mod, this->_silent_mode, this->_seq_mode, this->_read_fpga_info, &this->_tx_buf[0], &msg_id);
    if (auto res = this->_link->send(&this->_tx_buf[0], size); res.is_err()) return res;

    const auto mod_finished = ModSentFinished(mod);
    if (mod_finished && !wait_for_sent) return Ok(true);
    if (auto res = wait_msg_processed(msg_id); res.is_err() || mod_finished) return res;
  }
}

Error Controller::send(const core::SequencePtr& seq) {
  if (!this->is_open()) return Err(std::string("Link is not opened."));
  if (auto res = this->_stm->stop(); res.is_err()) return res;

  this->_seq_mode = true;
  while (true) {
    uint8_t msg_id;
    core::Logic::pack_header(core::COMMAND::SEQ_MODE, this->_silent_mode, this->_seq_mode, this->_read_fpga_info, &this->_tx_buf[0], &msg_id);
    size_t size;
    core::Logic::pack_body(seq, this->_geometry, &this->_tx_buf[0], &size);
    if (auto res = this->_link->send(&this->_tx_buf[0], size); res.is_err()) return res;

    if (SeqSentFinished(seq)) return wait_msg_processed(msg_id, 5000);
    if (auto res = wait_msg_processed(msg_id); res.is_err()) return res;
  }
}

Error Controller::set_output_delay(const std::vector<core::DataArray>& delay) const {
  if (!this->is_open()) return Err(std::string("Link is not opened."));
  if (delay.size() != this->_geometry->num_devices()) return Err(std::string("The number of devices is wrong."));

  uint8_t msg_id;
  core::Logic::pack_header(core::COMMAND::SET_DELAY, false, false, false, &this->_tx_buf[0], &msg_id);
  size_t size = 0;
  core::Logic::pack_delay_body(delay, &this->_tx_buf[0], &size);
  if (auto res = this->_link->send(&this->_tx_buf[0], size); res.is_err()) return res;
  return wait_msg_processed(msg_id, 200);
}

Result<std::vector<FirmwareInfo>, std::string> Controller::firmware_info_list() const {
  auto concat_byte = [](const uint8_t high, const uint16_t low) { return static_cast<uint16_t>(static_cast<uint16_t>(high) << 8 | low); };

  const auto num_devices = this->_geometry->num_devices();
  std::vector<uint16_t> cpu_versions(num_devices);
  if (auto res = send_header(core::COMMAND::READ_CPU_VER_LSB); res.is_err()) return Err(res.unwrap_err());
  for (size_t i = 0; i < num_devices; i++) cpu_versions[i] = _rx_buf[2 * i];
  if (auto res = send_header(core::COMMAND::READ_CPU_VER_MSB); res.is_err()) return Err(res.unwrap_err());
  for (size_t i = 0; i < num_devices; i++) cpu_versions[i] = concat_byte(_rx_buf[2 * i], cpu_versions[i]);

  std::vector<uint16_t> fpga_versions(num_devices);
  if (auto res = send_header(core::COMMAND::READ_FPGA_VER_LSB); res.is_err()) return Err(res.unwrap_err());
  for (size_t i = 0; i < num_devices; i++) fpga_versions[i] = _rx_buf[2 * i];
  if (auto res = send_header(core::COMMAND::READ_FPGA_VER_MSB); res.is_err()) return Err(res.unwrap_err());
  for (size_t i = 0; i < num_devices; i++) fpga_versions[i] = concat_byte(_rx_buf[2 * i], fpga_versions[i]);

  std::vector<FirmwareInfo> infos;
  for (size_t i = 0; i < num_devices; i++) infos.emplace_back(FirmwareInfo(static_cast<uint16_t>(i), cpu_versions[i], fpga_versions[i]));
  return Ok(std::move(infos));
}

std::shared_ptr<Controller::STMController> Controller::stm() const { return this->_stm; }

void Controller::STMController::add_gain(const core::GainPtr& gain) { _gains.emplace_back(gain); }
void Controller::STMController::add_gains(const std::vector<core::GainPtr>& gains) {
  for (const auto& g : gains) this->add_gain(g);
}

[[nodiscard]] Error Controller::STMController::start(const double freq) {
  auto len = this->_gains.size();
  auto interval_us = static_cast<uint32_t>(1000000. / static_cast<double>(freq) / static_cast<double>(len));
  this->_timer.SetInterval(interval_us);

  const auto current_size = this->_bodies.size();
  this->_bodies.resize(len);
  this->_sizes.resize(len);

  for (auto i = current_size; i < len; i++) {
    auto& g = this->_gains[i];
    if (auto res = g->build(this->_geometry); res.is_err()) return res;

    uint8_t msg_id = 0;
    this->_bodies[i] = std::make_unique<uint8_t[]>(this->_geometry->num_devices() * core::EC_OUTPUT_FRAME_SIZE);
    core::Logic::pack_header(nullptr, *this->_silent_mode, false, *this->_read_fpga_info, &this->_bodies[i][0], &msg_id);
    size_t size = 0;
    core::Logic::pack_body(g, &this->_bodies[i][0], &size);
    this->_sizes[i] = size;
  }

  size_t idx = 0;
  return this->_timer.start([this, idx, len]() mutable {
    if (auto expected = false; this->_lock.compare_exchange_weak(expected, true)) {
      const auto body_size = this->_sizes[idx];
      if (const auto res = this->_link->send(&this->_bodies[idx][0], body_size); res.is_err()) return;
      idx = (idx + 1) % len;
      this->_lock.store(false, std::memory_order_release);
    }
  });
}

[[nodiscard]] Error Controller::STMController::stop() { return this->_timer.stop(); }

[[nodiscard]] Error Controller::STMController::finish() {
  if (auto res = this->stop(); res.is_err()) return res;

  std::vector<core::GainPtr>().swap(this->_gains);
  std::vector<std::unique_ptr<uint8_t[]>>().swap(this->_bodies);
  std::vector<size_t>().swap(this->_sizes);

  return Ok(true);
}
}  // namespace autd
