// File: controller.cpp
// Project: lib
// Created Date: 05/11/2020
// Author: Shun Suzuki
// -----
// Last Modified: 19/06/2021
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

uint8_t Controller::ControllerProps::ctrl_flag() const {
  uint8_t flag = 0;
  if (this->_silent_mode) flag |= core::SILENT;
  if (this->_seq_mode) flag |= core::SEQ_MODE;
  if (this->_reads_fpga_info) flag |= core::READ_FPGA_INFO;
  if (this->_force_fan) flag |= core::FORCE_FAN;
  return flag;
}

ControllerPtr Controller::create() {
  struct Impl : Controller {
    Impl() : Controller() {}
  };
  return std::make_unique<Impl>();
}

bool Controller::is_open() const { return this->_link != nullptr && this->_link->is_open(); }

core::GeometryPtr& Controller::geometry() noexcept { return this->_geometry; }

bool& Controller::silent_mode() noexcept { return this->_props._silent_mode; }
bool& Controller::reads_fpga_info() noexcept { return this->_props._reads_fpga_info; }
bool& Controller::force_fan() noexcept { return this->_props._force_fan; }

Result<std::vector<uint8_t>, std::string> Controller::fpga_info() {
  const auto num_devices = this->_geometry->num_devices();
  if (auto res = this->_link->read(&_rx_buf[0], num_devices * core::EC_INPUT_FRAME_SIZE); res.is_err()) return Err(res.unwrap_err());
  for (size_t i = 0; i < num_devices; i++) this->_fpga_infos[i] = _rx_buf[2 * i];
  return Ok(_fpga_infos);
}

Error Controller::update_ctrl_flag() { return this->send(nullptr, nullptr); }

Error Controller::open(core::LinkPtr link) {
  if (is_open())
    if (auto close_res = this->close(); close_res.is_err()) return close_res;

  this->_tx_buf = std::make_unique<uint8_t[]>(this->_geometry->num_devices() * core::EC_OUTPUT_FRAME_SIZE);
  this->_rx_buf = std::make_unique<uint8_t[]>(this->_geometry->num_devices() * core::EC_INPUT_FRAME_SIZE);

  this->_fpga_infos.resize(this->_geometry->num_devices());
  this->_delay_en.resize(this->_geometry->num_devices());
  init_delay_en();

  this->_link = std::move(link);
  return this->_link->open();
}

void Controller::init_delay_en() {
  for (size_t dev = 0; dev < this->_geometry->num_devices(); dev++)
    for (size_t i = 0; i < core::NUM_TRANS_IN_UNIT; i++) this->_delay_en[dev][i] = 0xFF00;
}

Error Controller::clear() {
  this->init_delay_en();
  return send_header(core::COMMAND::CLEAR);
}

Error Controller::send_header(const core::COMMAND cmd) const {
  if (!this->is_open()) return Err(std::string("Link is not opened."));

  const auto send_size = sizeof(core::RxGlobalHeader);
  uint8_t msg_id = 0;
  core::Logic::pack_header(cmd, _props.ctrl_flag(), &_tx_buf[0], &msg_id);
  if (auto res = _link->send(&_tx_buf[0], send_size); res.is_err()) return res;
  return wait_msg_processed(msg_id);
}

Error Controller::send_delay_en() const {
  if (!this->is_open()) return Err(std::string("Link is not opened."));

  uint8_t msg_id;
  core::Logic::pack_header(core::COMMAND::SET_DELAY_EN, _props.ctrl_flag(), &this->_tx_buf[0], &msg_id);
  size_t size = 0;
  core::Logic::pack_delay_en_body(this->_delay_en, &this->_tx_buf[0], &size);
  if (auto res = this->_link->send(&this->_tx_buf[0], size); res.is_err()) return res;
  return wait_msg_processed(msg_id);
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
  if (auto res = this->stop(); res.is_err()) return res;
  if (auto res = this->clear(); res.is_err()) return res;

  auto res = this->_link->close();
  this->_link = nullptr;
  this->_tx_buf = nullptr;
  this->_rx_buf = nullptr;

  return res;
}

Error Controller::stop() const { return this->pause(); }
Error Controller::pause() const { return this->send_header(core::COMMAND::PAUSE); }
Error Controller::resume() const { return this->send_header(core::COMMAND::RESUME); }

Error Controller::send(const core::GainPtr& gain) { return this->send(gain, nullptr); }

Error Controller::send(const core::ModulationPtr& mod) { return this->send(nullptr, mod); }

Error Controller::send(const core::GainPtr& gain, const core::ModulationPtr& mod) {
  if (!this->is_open()) return Err(std::string("Link is not opened."));

  if (mod != nullptr)
    if (auto res = mod->build(); res.is_err()) return res;

  if (gain != nullptr) {
    this->_props._seq_mode = false;
    if (auto res = gain->build(this->_geometry); res.is_err()) return res;
  }

  size_t size = 0;
  core::Logic::pack_body(gain, &this->_tx_buf[0], &size);

  auto mod_finished = [](const core::ModulationPtr& m) { return m == nullptr || m->sent() == m->buffer().size(); };
  while (true) {
    uint8_t msg_id = 0;
    core::Logic::pack_header(mod, _props.ctrl_flag(), &this->_tx_buf[0], &msg_id);
    if (auto res = this->_link->send(&this->_tx_buf[0], size); res.is_err()) return res;
    if (auto res = wait_msg_processed(msg_id); res.is_err() || mod_finished(mod)) return res;
  }
}

Error Controller::send(const core::SequencePtr& seq) {
  if (!this->is_open()) return Err(std::string("Link is not opened."));

  auto seq_finished = [](const core::SequencePtr& s) { return s == nullptr || s->sent() == s->control_points().size(); };

  this->_props._seq_mode = true;
  while (true) {
    uint8_t msg_id;
    core::Logic::pack_header(core::COMMAND::SEQ_MODE, _props.ctrl_flag(), &this->_tx_buf[0], &msg_id);
    size_t size;
    core::Logic::pack_body(seq, this->_geometry, &this->_tx_buf[0], &size);
    if (auto res = this->_link->send(&this->_tx_buf[0], size); res.is_err()) return res;
    if (auto res = wait_msg_processed(msg_id); res.is_err() || seq_finished(seq)) return res;
  }
}

Error Controller::set_output_delay(const std::vector<std::array<uint8_t, core::NUM_TRANS_IN_UNIT>>& delay) {
  if (delay.size() != this->_geometry->num_devices()) return Err(std::string("The number of devices is wrong."));

  for (size_t dev = 0; dev < this->_geometry->num_devices(); dev++)
    for (size_t i = 0; i < core::NUM_TRANS_IN_UNIT; i++) this->_delay_en[dev][i] = (this->_delay_en[dev][i] & 0xFF00) | delay[dev][i];

  return this->send_delay_en();
}

Error Controller::set_enable(const std::vector<std::array<bool, core::NUM_TRANS_IN_UNIT>>& enable) {
  if (enable.size() != this->_geometry->num_devices()) return Err(std::string("The number of devices is wrong."));

  for (size_t dev = 0; dev < this->_geometry->num_devices(); dev++)
    for (size_t i = 0; i < core::NUM_TRANS_IN_UNIT; i++)
      this->_delay_en[dev][i] = (this->_delay_en[dev][i] & 0x00FF) | (enable[dev][i] ? 0xFF00 : 0x0000);

  return this->send_delay_en();
}

Error Controller::set_enable(const std::vector<std::array<uint8_t, core::NUM_TRANS_IN_UNIT>>& enable) {
  if (enable.size() != this->_geometry->num_devices()) return Err(std::string("The number of devices is wrong."));

  for (size_t dev = 0; dev < this->_geometry->num_devices(); dev++)
    for (size_t i = 0; i < core::NUM_TRANS_IN_UNIT; i++) this->_delay_en[dev][i] = (this->_delay_en[dev][i] & 0x00FF) | enable[dev][i];

  return this->send_delay_en();
}

Error Controller::set_delay_enable(const std::vector<std::array<uint16_t, core::NUM_TRANS_IN_UNIT>>& delay_enable) {
  if (delay_enable.size() != this->_geometry->num_devices()) return Err(std::string("The number of devices is wrong."));

  for (size_t dev = 0; dev < this->_geometry->num_devices(); dev++)
    std::memcpy(&this->_delay_en[dev][0], &delay_enable[dev][0], sizeof(uint16_t) * core::NUM_TRANS_IN_UNIT);

  return this->send_delay_en();
}

Result<std::vector<FirmwareInfo>, std::string> Controller::firmware_info_list() const {
  auto concat_byte = [](const uint8_t high, const uint16_t low) { return static_cast<uint16_t>(static_cast<uint16_t>(high) << 8 | low); };

  const auto num_devices = this->_geometry->num_devices();
  std::vector<uint16_t> cpu_versions(num_devices);
  if (auto res = send_header(core::COMMAND::READ_CPU_VER_LSB); res.is_err()) return Err(res.unwrap_err());
  for (size_t i = 0; i < num_devices; i++) cpu_versions[i] = this->_rx_buf[2 * i];
  if (auto res = send_header(core::COMMAND::READ_CPU_VER_MSB); res.is_err()) return Err(res.unwrap_err());
  for (size_t i = 0; i < num_devices; i++) cpu_versions[i] = concat_byte(this->_rx_buf[2 * i], cpu_versions[i]);

  std::vector<uint16_t> fpga_versions(num_devices);
  if (auto res = send_header(core::COMMAND::READ_FPGA_VER_LSB); res.is_err()) return Err(res.unwrap_err());
  for (size_t i = 0; i < num_devices; i++) fpga_versions[i] = this->_rx_buf[2 * i];
  if (auto res = send_header(core::COMMAND::READ_FPGA_VER_MSB); res.is_err()) return Err(res.unwrap_err());
  for (size_t i = 0; i < num_devices; i++) fpga_versions[i] = concat_byte(this->_rx_buf[2 * i], fpga_versions[i]);

  std::vector<FirmwareInfo> infos;
  for (size_t i = 0; i < num_devices; i++) infos.emplace_back(FirmwareInfo(static_cast<uint16_t>(i), cpu_versions[i], fpga_versions[i]));
  return Ok(std::move(infos));
}

std::unique_ptr<Controller::STMController> Controller::stm() {
  struct Impl : STMController {
    Impl(std::unique_ptr<STMTimerCallback> callback, Controller* p_cnt) : STMController(p_cnt, std::move(callback)) {}
  };
  return std::make_unique<Impl>(std::make_unique<STMTimerCallback>(std::move(this->_link)), this);
}

Error Controller::STMController::add_gain(const core::GainPtr& gain) const {
  if (auto res = gain->build(this->_p_cnt->_geometry); res.is_err()) return res;

  uint8_t msg_id = 0;
  auto build_buf = std::make_unique<uint8_t[]>(this->_p_cnt->_geometry->num_devices() * core::EC_OUTPUT_FRAME_SIZE);
  core::Logic::pack_header(nullptr, this->_p_cnt->_props.ctrl_flag(), &build_buf[0], &msg_id);
  size_t size = 0;
  core::Logic::pack_body(gain, &build_buf[0], &size);

  this->_handler->add(std::move(build_buf), size);
  return Ok(true);
}

Error Controller::STMController::start(const double freq) {
  if (this->_handler == nullptr) return Err(std::string("STM has been already started"));

  const auto len = this->_handler->_bodies.size();
  const auto interval_us = static_cast<uint32_t>(1000000. / static_cast<double>(freq) / static_cast<double>(len));
  auto res = core::Timer<STMTimerCallback>::start(std::move(this->_handler), interval_us);
  this->_handler = nullptr;
  if (res.is_err()) return Err(res.unwrap_err());
  this->_timer = res.unwrap();
  return Ok(true);
}

Error Controller::STMController::finish() {
  if (auto res = this->stop(); res.is_err()) return Err(res.unwrap_err());
  this->_handler->clear();
  this->_p_cnt->_link = std::move(this->_handler->_link);
  return Ok(true);
}

Error Controller::STMController::stop() {
  if (this->_handler != nullptr) return Ok(true);
  auto res = this->_timer->stop();
  if (res.is_err()) return Err(res.unwrap_err());
  this->_handler = res.unwrap();
  return Ok(true);
}

}  // namespace autd
