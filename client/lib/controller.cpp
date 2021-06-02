// File: controller.cpp
// Project: lib
// Created Date: 05/11/2020
// Author: Shun Suzuki
// -----
// Last Modified: 02/06/2021
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

std::unique_ptr<Controller> Controller::create() {
  struct Impl : Controller {
    Impl() : Controller() {}
  };
  return std::make_unique<Impl>();
}

bool Controller::is_open() const { return this->_link != nullptr && this->_link->is_open(); }

core::GeometryPtr& Controller::geometry() noexcept { return this->_props._geometry; }

bool& Controller::silent_mode() noexcept { return this->_props._silent_mode; }
bool& Controller::reads_fpga_info() noexcept { return this->_props._reads_fpga_info; }
bool& Controller::force_fan() noexcept { return this->_props._force_fan; }

Result<std::vector<uint8_t>, std::string> Controller::fpga_info() {
  const auto num_devices = this->_props._geometry->num_devices();
  this->_fpga_infos.resize(num_devices);
  if (auto res = this->_link->read(&_props._rx_buf[0], num_devices * core::EC_INPUT_FRAME_SIZE); res.is_err()) return Err(res.unwrap_err());
  for (size_t i = 0; i < num_devices; i++) this->_fpga_infos[i] = _props._rx_buf[2 * i];
  return Ok(_fpga_infos);
}

Error Controller::update_ctrl_flag() { return this->send(nullptr, nullptr); }

Error Controller::open(core::LinkPtr link) {
  if (is_open())
    if (auto close_res = this->close(); close_res.is_err()) return close_res;

  this->_props._tx_buf = std::make_unique<uint8_t[]>(this->_props._geometry->num_devices() * core::EC_OUTPUT_FRAME_SIZE);
  this->_props._rx_buf = std::make_unique<uint8_t[]>(this->_props._geometry->num_devices() * core::EC_INPUT_FRAME_SIZE);

  this->_link = std::move(link);
  return this->_link->open();
}

Error Controller::synchronize(const core::Configuration config) {
  if (!this->is_open()) return Err(std::string("Link is not opened."));

  this->_props._config = config;
  uint8_t msg_id = 0;
  core::Logic::pack_header(core::COMMAND::INIT_MOD_CLOCK, _props.ctrl_flag(), &_props._tx_buf[0], &msg_id);
  size_t size = 0;
  const auto num_devices = this->_props._geometry->num_devices();
  core::Logic::pack_sync_body(config, num_devices, &_props._tx_buf[0], &size);

  if (auto res = this->_link->send(&_props._tx_buf[0], size); res.is_err()) return res;
  return wait_msg_processed(msg_id, 5000);
}

Error Controller::clear() const { return send_header(core::COMMAND::CLEAR); }

Error Controller::send_header(const core::COMMAND cmd) const {
  if (!this->is_open()) return Err(std::string("Link is not opened."));

  const auto send_size = sizeof(core::RxGlobalHeader);
  uint8_t msg_id = 0;
  core::Logic::pack_header(cmd, _props.ctrl_flag(), &_props._tx_buf[0], &msg_id);
  if (auto res = _link->send(&_props._tx_buf[0], send_size); res.is_err()) return res;
  return wait_msg_processed(msg_id);
}

Error Controller::wait_msg_processed(const uint8_t msg_id, const size_t max_trial) const {
  const auto num_devices = this->_props._geometry->num_devices();
  const auto buffer_len = num_devices * core::EC_INPUT_FRAME_SIZE;
  for (size_t i = 0; i < max_trial; i++) {
    auto res = this->_link->read(&_props._rx_buf[0], buffer_len);
    if (res.is_err()) return res;
    if (!res.unwrap()) continue;
    if (core::Logic::is_msg_processed(num_devices, msg_id, &_props._rx_buf[0])) return Ok(true);

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
  this->_props._tx_buf = nullptr;
  this->_props._rx_buf = nullptr;

  return res;
}

Error Controller::stop() { return this->send(gain::NullGain::create(), nullptr); }

Error Controller::send(const core::GainPtr& gain) { return this->send(gain, nullptr); }

Error Controller::send(const core::ModulationPtr& mod) { return this->send(nullptr, mod); }

Error Controller::send(const core::GainPtr& gain, const core::ModulationPtr& mod) {
  if (!this->is_open()) return Err(std::string("Link is not opened."));

  if (mod != nullptr)
    if (auto res = mod->build(this->_props._config); res.is_err()) return res;

  if (gain != nullptr) {
    this->_props._seq_mode = false;
    if (auto res = gain->build(this->_props._geometry); res.is_err()) return res;
  }

  size_t size = 0;
  core::Logic::pack_body(gain, &this->_props._tx_buf[0], &size);

  auto mod_finished = [](const core::ModulationPtr& mod) { return mod == nullptr || mod->sent() == mod->buffer().size(); };
  while (true) {
    uint8_t msg_id = 0;
    core::Logic::pack_header(mod, _props.ctrl_flag(), &this->_props._tx_buf[0], &msg_id);
    if (auto res = this->_link->send(&this->_props._tx_buf[0], size); res.is_err()) return res;
    if (auto res = wait_msg_processed(msg_id); res.is_err() || mod_finished(mod)) return res;
  }
}

Error Controller::send(const core::SequencePtr& seq) {
  if (!this->is_open()) return Err(std::string("Link is not opened."));

  auto seq_finished = [](const core::SequencePtr& seq) { return seq == nullptr || seq->sent() == seq->control_points().size(); };

  this->_props._seq_mode = true;
  while (true) {
    uint8_t msg_id;
    core::Logic::pack_header(core::COMMAND::SEQ_MODE, _props.ctrl_flag(), &this->_props._tx_buf[0], &msg_id);
    size_t size;
    core::Logic::pack_body(seq, this->_props._geometry, &this->_props._tx_buf[0], &size);
    if (auto res = this->_link->send(&this->_props._tx_buf[0], size); res.is_err()) return res;
    if (auto res = wait_msg_processed(msg_id); res.is_err() || seq_finished(seq)) return res;
  }
}

Error Controller::set_output_delay(const std::vector<core::DataArray>& delay) const {
  if (!this->is_open()) return Err(std::string("Link is not opened."));
  if (delay.size() != this->_props._geometry->num_devices()) return Err(std::string("The number of devices is wrong."));

  uint8_t msg_id;
  core::Logic::pack_header(core::COMMAND::SET_DELAY, _props.ctrl_flag(), &this->_props._tx_buf[0], &msg_id);
  size_t size = 0;
  core::Logic::pack_delay_body(delay, &this->_props._tx_buf[0], &size);
  if (auto res = this->_link->send(&this->_props._tx_buf[0], size); res.is_err()) return res;
  return wait_msg_processed(msg_id);
}

Result<std::vector<FirmwareInfo>, std::string> Controller::firmware_info_list() const {
  auto concat_byte = [](const uint8_t high, const uint16_t low) { return static_cast<uint16_t>(static_cast<uint16_t>(high) << 8 | low); };

  const auto num_devices = this->_props._geometry->num_devices();
  std::vector<uint16_t> cpu_versions(num_devices);
  if (auto res = send_header(core::COMMAND::READ_CPU_VER_LSB); res.is_err()) return Err(res.unwrap_err());
  for (size_t i = 0; i < num_devices; i++) cpu_versions[i] = this->_props._rx_buf[2 * i];
  if (auto res = send_header(core::COMMAND::READ_CPU_VER_MSB); res.is_err()) return Err(res.unwrap_err());
  for (size_t i = 0; i < num_devices; i++) cpu_versions[i] = concat_byte(this->_props._rx_buf[2 * i], cpu_versions[i]);

  std::vector<uint16_t> fpga_versions(num_devices);
  if (auto res = send_header(core::COMMAND::READ_FPGA_VER_LSB); res.is_err()) return Err(res.unwrap_err());
  for (size_t i = 0; i < num_devices; i++) fpga_versions[i] = this->_props._rx_buf[2 * i];
  if (auto res = send_header(core::COMMAND::READ_FPGA_VER_MSB); res.is_err()) return Err(res.unwrap_err());
  for (size_t i = 0; i < num_devices; i++) fpga_versions[i] = concat_byte(this->_props._rx_buf[2 * i], fpga_versions[i]);

  std::vector<FirmwareInfo> infos;
  for (size_t i = 0; i < num_devices; i++) infos.emplace_back(FirmwareInfo(static_cast<uint16_t>(i), cpu_versions[i], fpga_versions[i]));
  return Ok(std::move(infos));
}

std::unique_ptr<Controller::STMController> Controller::stm() {
  struct Impl : STMController {
    Impl(std::unique_ptr<STMTimerCallback> callback, ControllerProps props) : STMController(std::move(callback), std::move(props)) {}
  };
  return std::make_unique<Impl>(std::make_unique<STMTimerCallback>(std::move(this->_link)), std::move(this->_props));
}

std::unique_ptr<Controller> Controller::STMController::controller() {
  struct Impl : Controller {
    Impl(core::LinkPtr link, ControllerProps props) : Controller(std::move(link), std::move(props)) {}
  };
  return std::make_unique<Impl>(std::move(this->_handler->_link), std::move(this->_props));
}

Error Controller::STMController::add_gain(const core::GainPtr& gain) const {
  if (auto res = gain->build(this->_props._geometry); res.is_err()) return res;

  uint8_t msg_id = 0;
  auto build_buf = std::make_unique<uint8_t[]>(this->_props._geometry->num_devices() * core::EC_OUTPUT_FRAME_SIZE);
  core::Logic::pack_header(nullptr, this->_props.ctrl_flag(), &build_buf[0], &msg_id);
  size_t size = 0;
  core::Logic::pack_body(gain, &build_buf[0], &size);

  this->_handler->add(std::move(build_buf), size);
  return Ok(true);
}

Result<std::unique_ptr<Controller::STMTimer>, std::string> Controller::STMController::start(const double freq) {
  const auto len = this->_handler->_bodies.size();
  const auto interval_us = static_cast<uint32_t>(1000000. / static_cast<double>(freq) / static_cast<double>(len));
  auto res = core::Timer<STMTimerCallback>::start(std::move(this->_handler), interval_us);
  if (res.is_err()) return Err(res.unwrap_err());

  struct Impl : STMTimer {
    Impl(std::unique_ptr<core::Timer<STMTimerCallback>> timer, ControllerProps props) : STMTimer(std::move(timer), std::move(props)) {}
  };
  std::unique_ptr<STMTimer> cnt = std::make_unique<Impl>(res.unwrap(), std::move(this->_props));
  return Ok(std::move(cnt));
}

void Controller::STMController::finish() const { this->_handler->clear(); }

Result<std::unique_ptr<Controller::STMController>, std::string> Controller::STMTimer::stop() {
  auto res = this->_timer->stop();
  if (res.is_err()) return Err(res.unwrap_err());
  struct Impl : STMController {
    Impl(std::unique_ptr<STMTimerCallback> handler, ControllerProps props) : STMController(std::move(handler), std::move(props)) {}
  };
  std::unique_ptr<STMController> cnt = std::make_unique<Impl>(res.unwrap(), std::move(this->_props));
  return Ok(std::move(cnt));
}

}  // namespace autd
