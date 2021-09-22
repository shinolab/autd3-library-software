// File: controller.cpp
// Project: lib
// Created Date: 05/11/2020
// Author: Shun Suzuki
// -----
// Last Modified: 22/09/2021
// Modified By: Shun Suzuki (suzuki@hapis.k.u-tokyo.ac.jp)
// -----
// Copyright (c) 2021 Hapis Lab. All rights reserved.
//

#include "autd3/controller.hpp"

#include <condition_variable>
#include <vector>

#include "autd3/core/ec_config.hpp"
#include "autd3/core/logic.hpp"
#include "autd3/gain/primitive.hpp"

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

const std::vector<uint8_t>& Controller::fpga_info() {
  const auto num_devices = this->_geometry->num_devices();
  this->_link->read(&_rx_buf[0], num_devices * core::EC_INPUT_FRAME_SIZE);
  for (size_t i = 0; i < num_devices; i++) this->_fpga_infos[i] = _rx_buf[2 * i];
  return _fpga_infos;
}

bool Controller::update_ctrl_flag() { return this->send(nullptr, nullptr, true); }

void Controller::open(core::LinkPtr link) {
  if (is_open()) this->close();

  this->_tx_buf = std::make_unique<uint8_t[]>(this->_geometry->num_devices() * core::EC_OUTPUT_FRAME_SIZE);
  this->_rx_buf = std::make_unique<uint8_t[]>(this->_geometry->num_devices() * core::EC_INPUT_FRAME_SIZE);

  this->_fpga_infos.resize(this->_geometry->num_devices());
  this->_delay.resize(this->_geometry->num_devices());
  this->_offset.resize(this->_geometry->num_devices());
  init_delay_offset();

  this->_link = std::move(link);
  this->_link->open();
}

void Controller::init_delay_offset() {
  for (size_t dev = 0; dev < this->_geometry->num_devices(); dev++) {
    std::memset(&this->_delay[dev][0], 0x00, core::NUM_TRANS_IN_UNIT);
    std::memset(&this->_offset[dev][0], 0xFF, core::NUM_TRANS_IN_UNIT);
  }
}

bool Controller::clear() {
  this->init_delay_offset();
  return send_header(core::COMMAND::CLEAR);
}

bool Controller::send_header(const core::COMMAND cmd) const {
  constexpr auto send_size = sizeof(core::RxGlobalHeader);
  uint8_t msg_id = 0;
  core::Logic::pack_header(cmd, _props.ctrl_flag(), &_tx_buf[0], &msg_id);
  _link->send(&_tx_buf[0], send_size);
  return wait_msg_processed(msg_id);
}

bool Controller::send_delay_offset() const {
  uint8_t msg_id;
  core::Logic::pack_header(core::COMMAND::SET_DELAY_OFFSET, _props.ctrl_flag(), &this->_tx_buf[0], &msg_id);
  size_t size = 0;
  core::Logic::pack_delay_offset_body(this->_delay, this->_offset, &this->_tx_buf[0], &size);
  this->_link->send(&this->_tx_buf[0], size);
  return wait_msg_processed(msg_id);
}

bool Controller::wait_msg_processed(const uint8_t msg_id, const size_t max_trial) const {
  const auto num_devices = this->_geometry->num_devices();
  const auto buffer_len = num_devices * core::EC_INPUT_FRAME_SIZE;
  for (size_t i = 0; i < max_trial; i++) {
    this->_link->read(&_rx_buf[0], buffer_len);
    if (core::Logic::is_msg_processed(num_devices, msg_id, &_rx_buf[0])) return true;

    auto wait = static_cast<size_t>(std::ceil(core::EC_TRAFFIC_DELAY * 1000.0 / core::EC_DEVICE_PER_FRAME * static_cast<double>(num_devices)));
    std::this_thread::sleep_for(std::chrono::milliseconds(wait));
  }

  return false;
}

bool Controller::close() {
  if (this->_link == nullptr) return true;
  if (!this->stop()) return false;
  if (!this->clear()) return false;

  this->_link->close();
  this->_link = nullptr;
  this->_tx_buf = nullptr;
  this->_rx_buf = nullptr;

  return true;
}

bool Controller::stop() {
  // To suppress shutdown noise
  const auto silent = this->silent_mode();
  this->silent_mode() = true;
  const auto res = this->send(gain::NullGain::create());
  this->silent_mode() = silent;

  return this->pause() && res;
}

bool Controller::pause() const { return this->send_header(core::COMMAND::PAUSE); }
bool Controller::resume() const { return this->send_header(core::COMMAND::RESUME); }

bool Controller::send(const core::GainPtr& gain, const bool wait_for_msg_processed) { return this->send(gain, nullptr, wait_for_msg_processed); }

bool Controller::send(const core::ModulationPtr& mod) { return this->send(nullptr, mod, true); }

bool Controller::send(const core::GainPtr& gain, const core::ModulationPtr& mod, bool wait_for_msg_processed) {
  if (mod != nullptr) {
    mod->build();
    wait_for_msg_processed = true;
  }
  if (gain != nullptr) {
    this->_props._seq_mode = false;
    gain->build(this->_geometry);
  }

  size_t size = 0;
  core::Logic::pack_body(gain, &this->_tx_buf[0], &size);

  auto mod_finished = [](const core::ModulationPtr& m) { return m == nullptr || m->sent() == m->buffer().size(); };
  while (true) {
    uint8_t msg_id = 0;
    core::Logic::pack_header(mod, _props.ctrl_flag(), &this->_tx_buf[0], &msg_id);
    this->_link->send(&this->_tx_buf[0], size);
    if (!wait_for_msg_processed && mod_finished(mod)) return false;
    if (const auto res = wait_msg_processed(msg_id); !res || mod_finished(mod)) return res;
  }
}

bool Controller::send(const core::PointSequencePtr& seq) {
  auto seq_finished = [](const core::PointSequencePtr& s) { return s == nullptr || s->sent() == s->control_points().size(); };

  this->_props._seq_mode = true;
  while (true) {
    uint8_t msg_id;
    core::Logic::pack_header(core::COMMAND::SEQ_FOCI_MODE, _props.ctrl_flag(), &this->_tx_buf[0], &msg_id);
    size_t size;
    core::Logic::pack_body(seq, this->_geometry, &this->_tx_buf[0], &size);
    this->_link->send(&this->_tx_buf[0], size);
    if (const auto res = wait_msg_processed(msg_id); !res || seq_finished(seq)) return res;
  }
}

bool Controller::send(const core::GainSequencePtr& seq) {
  auto seq_finished = [](const core::GainSequencePtr& s) { return s == nullptr || s->sent() >= s->gains().size() + 1; };

  for (auto&& g : seq->gains()) g->build(this->_geometry);

  this->_props._seq_mode = true;
  while (true) {
    uint8_t msg_id;
    core::Logic::pack_header(core::COMMAND::SEQ_GAIN_MODE, _props.ctrl_flag(), &this->_tx_buf[0], &msg_id);
    size_t size;
    core::Logic::pack_body(seq, this->_geometry, &this->_tx_buf[0], &size);
    this->_link->send(&this->_tx_buf[0], size);
    if (const auto res = wait_msg_processed(msg_id); !res || seq_finished(seq)) return res;
  }
}

bool Controller::set_output_delay(const std::vector<std::array<uint8_t, core::NUM_TRANS_IN_UNIT>>& delay) {
  if (delay.size() != this->_geometry->num_devices()) throw core::exception::SetOutputConfigError("The number of devices is wrong");

  for (size_t dev = 0; dev < this->_geometry->num_devices(); dev++) std::memcpy(&this->_delay[dev][0], &delay[dev][0], core::NUM_TRANS_IN_UNIT);

  return this->send_delay_offset();
}

bool Controller::set_duty_offset(const std::vector<std::array<uint8_t, core::NUM_TRANS_IN_UNIT>>& offset) {
  if (offset.size() != this->_geometry->num_devices()) throw core::exception::SetOutputConfigError("The number of devices is wrong");

  for (size_t dev = 0; dev < this->_geometry->num_devices(); dev++) std::memcpy(&this->_offset[dev][0], &offset[dev][0], core::NUM_TRANS_IN_UNIT);

  return this->send_delay_offset();
}

bool Controller::set_delay_offset(const std::vector<std::array<uint8_t, core::NUM_TRANS_IN_UNIT>>& delay,
                                  const std::vector<std::array<uint8_t, core::NUM_TRANS_IN_UNIT>>& offset) {
  if (delay.size() != this->_geometry->num_devices() || offset.size() != this->_geometry->num_devices())
    throw core::exception::SetOutputConfigError("The number of devices is wrong");

  for (size_t dev = 0; dev < this->_geometry->num_devices(); dev++) {
    std::memcpy(&this->_delay[dev][0], &delay[dev][0], core::NUM_TRANS_IN_UNIT);
    std::memcpy(&this->_offset[dev][0], &offset[dev][0], core::NUM_TRANS_IN_UNIT);
  }

  return this->send_delay_offset();
}

std::vector<FirmwareInfo> Controller::firmware_info_list() const {
  auto concat_byte = [](const uint8_t high, const uint16_t low) { return static_cast<uint16_t>(static_cast<uint16_t>(high) << 8 | low); };

  std::vector<FirmwareInfo> infos;

  const auto num_devices = this->_geometry->num_devices();
  std::vector<uint16_t> cpu_versions(num_devices);
  if (const auto res = send_header(core::COMMAND::READ_CPU_VER_LSB); !res) return infos;
  for (size_t i = 0; i < num_devices; i++) cpu_versions[i] = this->_rx_buf[2 * i];
  if (const auto res = send_header(core::COMMAND::READ_CPU_VER_MSB); !res) return infos;
  for (size_t i = 0; i < num_devices; i++) cpu_versions[i] = concat_byte(this->_rx_buf[2 * i], cpu_versions[i]);

  std::vector<uint16_t> fpga_versions(num_devices);
  if (const auto res = send_header(core::COMMAND::READ_FPGA_VER_LSB); !res) return infos;
  for (size_t i = 0; i < num_devices; i++) fpga_versions[i] = this->_rx_buf[2 * i];
  if (const auto res = send_header(core::COMMAND::READ_FPGA_VER_MSB); !res) return infos;
  for (size_t i = 0; i < num_devices; i++) fpga_versions[i] = concat_byte(this->_rx_buf[2 * i], fpga_versions[i]);

  for (size_t i = 0; i < num_devices; i++) infos.emplace_back(FirmwareInfo(static_cast<uint16_t>(i), cpu_versions[i], fpga_versions[i]));
  return infos;
}

std::unique_ptr<Controller::STMController> Controller::stm() {
  struct Impl : STMController {
    Impl(std::unique_ptr<STMTimerCallback> callback, Controller* p_cnt) : STMController(p_cnt, std::move(callback)) {}
  };
  return std::make_unique<Impl>(std::make_unique<STMTimerCallback>(std::move(this->_link)), this);
}

void Controller::STMController::add_gain(const core::GainPtr& gain) const {
  gain->build(this->_p_cnt->_geometry);

  uint8_t msg_id = 0;
  auto build_buf = std::make_unique<uint8_t[]>(this->_p_cnt->_geometry->num_devices() * core::EC_OUTPUT_FRAME_SIZE);
  core::Logic::pack_header(nullptr, this->_p_cnt->_props.ctrl_flag(), &build_buf[0], &msg_id);
  size_t size = 0;
  core::Logic::pack_body(gain, &build_buf[0], &size);

  this->_handler->add(std::move(build_buf), size);
}

void Controller::STMController::start(const double freq) {
  if (this->_handler == nullptr) throw core::exception::STMError("STM has been already started");

  const auto len = this->_handler->_bodies.size();
  const auto interval_us = static_cast<uint32_t>(1000000. / static_cast<double>(freq) / static_cast<double>(len));
  this->_timer = core::Timer<STMTimerCallback>::start(std::move(this->_handler), interval_us);
  this->_handler = nullptr;
}

void Controller::STMController::finish() {
  this->stop();
  this->_handler->clear();
  this->_p_cnt->_link = std::move(this->_handler->_link);
}

void Controller::STMController::stop() {
  if (this->_handler == nullptr) this->_handler = this->_timer->stop();
}

void Controller::STMTimerCallback::add(std::unique_ptr<uint8_t[]> data, const size_t size) {
  this->_bodies.emplace_back(std::move(data));
  this->_sizes.emplace_back(size);
}
void Controller::STMTimerCallback::clear() {
  std::vector<std::unique_ptr<uint8_t[]>>().swap(this->_bodies);
  std::vector<size_t>().swap(this->_sizes);
  this->_idx = 0;
}

void Controller::STMTimerCallback::callback() {
  if (auto expected = false; _lock.compare_exchange_weak(expected, true)) {
    this->_link->send(&this->_bodies[_idx][0], this->_sizes[_idx]);
    this->_idx = (this->_idx + 1) % this->_bodies.size();
    _lock.store(false, std::memory_order_release);
  }
}

}  // namespace autd
