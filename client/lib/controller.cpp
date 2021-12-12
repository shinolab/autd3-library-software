// File: controller.cpp
// Project: lib
// Created Date: 05/11/2020
// Author: Shun Suzuki
// -----
// Last Modified: 12/12/2021
// Modified By: Shun Suzuki (suzuki@hapis.k.u-tokyo.ac.jp)
// -----
// Copyright (c) 2021 Hapis Lab. All rights reserved.
//

#include "autd3/controller.hpp"

#include <condition_variable>
#include <vector>

#include "autd3/core/common_header.hpp"
#include "autd3/core/ec_config.hpp"
#include "autd3/gain/primitive.hpp"

namespace autd {

uint8_t Controller::ControllerProps::fpga_ctrl_flag() const {
  uint8_t flag = 0;
  if (this->_output_enable) flag |= core::OUTPUT_ENABLE;
  if (this->_output_balance) flag |= core::OUTPUT_BALANCE;
  if (this->_silent_mode) flag |= core::SILENT;
  if (this->_force_fan) flag |= core::FORCE_FAN;
  return flag;
}

uint8_t Controller::ControllerProps::cpu_ctrl_flag() const {
  uint8_t flag = 0;
  if (this->_reads_fpga_info) flag |= core::READS_FPGA_INFO;
  return flag;
}

Controller::~Controller() noexcept {
  try {
    this->close();
  } catch (...) {
  }
}

bool Controller::is_open() const { return this->_link != nullptr && this->_link->is_open(); }

core::Geometry& Controller::geometry() noexcept { return this->_geometry; }
const core::Geometry& Controller::geometry() const noexcept { return this->_geometry; }

bool& Controller::output_enable() noexcept { return this->_props._output_enable; }
bool& Controller::silent_mode() noexcept { return this->_props._silent_mode; }
bool& Controller::reads_fpga_info() noexcept { return this->_props._reads_fpga_info; }
bool& Controller::force_fan() noexcept { return this->_props._force_fan; }
bool& Controller::output_balance() noexcept { return this->_props._output_balance; }
bool& Controller::check_ack() noexcept { return this->_check_ack; }

bool Controller::output_enable() const noexcept { return this->_props._output_enable; }
bool Controller::silent_mode() const noexcept { return this->_props._silent_mode; }
bool Controller::reads_fpga_info() const noexcept { return this->_props._reads_fpga_info; }
bool Controller::force_fan() const noexcept { return this->_props._force_fan; }
bool Controller::output_balance() const noexcept { return this->_props._output_balance; }
bool Controller::check_ack() const noexcept { return this->_check_ack; }

const std::vector<uint8_t>& Controller::fpga_info() {
  const auto num_devices = this->_geometry.num_devices();
  this->_link->receive(this->_rx_buf);
  for (size_t i = 0; i < num_devices; i++) this->_fpga_infos[i] = _rx_buf[i].ack;
  return _fpga_infos;
}

bool Controller::update_ctrl_flag() {
  core::CommonHeader header;
  return send(header);
}

void Controller::open(core::LinkPtr link) {
  this->close();

  this->_tx_buf = core::TxDatagram(this->_geometry.num_devices());
  this->_rx_buf = core::RxDatagram(this->_geometry.num_devices());

  this->_fpga_infos.resize(this->_geometry.num_transducers());
  this->_delay_offset.resize(this->_geometry.num_transducers());

  link->open();
  this->_link = std::move(link);
}

bool Controller::clear() {
  core::SpecialMessageIdHeader header(core::MSG_CLEAR);
  return send(header);
}

bool Controller::wait_msg_processed(const uint8_t msg_id, const size_t max_trial) {
  if (!this->_check_ack) return true;
  const auto num_devices = this->_geometry.num_devices();
  for (size_t i = 0; i < max_trial; i++) {
    this->_link->receive(this->_rx_buf);
    if (core::is_msg_processed(num_devices, msg_id, _rx_buf)) return true;
    auto wait = static_cast<size_t>(std::ceil(core::EC_TRAFFIC_DELAY * 1000.0 / core::EC_DEVICE_PER_FRAME * static_cast<double>(num_devices)));
    std::this_thread::sleep_for(std::chrono::milliseconds(wait));
  }
  return false;
}

bool Controller::close() {
  if (!this->is_open()) return true;
  if (!this->stop()) return false;
  if (!this->clear()) return false;

  this->_link->close();
  this->_link = nullptr;

  return true;
}

bool Controller::stop() {
  // To suppress shutdown noise
  const auto silent = this->silent_mode();
  this->silent_mode() = true;
  gain::Null g;
  const auto res = this->send(g);
  this->silent_mode() = silent;
  return res;
}

bool Controller::pause() {
  this->_props._output_enable = false;
  return this->update_ctrl_flag();
}

bool Controller::resume() {
  this->_props._output_enable = true;
  return this->update_ctrl_flag();
}

bool Controller::send(core::IDatagramHeader& header) {
  header.init();
  const auto msg_id = header.pack(this->_geometry, _tx_buf, _props.fpga_ctrl_flag(), _props.cpu_ctrl_flag());
  this->_link->send(this->_tx_buf);
  return wait_msg_processed(msg_id);
}

bool Controller::send(core::IDatagramBody& body) {
  body.init();
  const auto msg_id = body.pack(this->_geometry, _tx_buf, _props.fpga_ctrl_flag(), _props.cpu_ctrl_flag());
  this->_link->send(this->_tx_buf);
  return wait_msg_processed(msg_id);
}

//
// bool Controller::send(core::IDatagram& gain) {
//  this->_props._output_enable = true;
//
//  gain.init();
//
//  const auto msg_id = gain.prepare(_props.fpga_ctrl_flag(), _props.cpu_ctrl_flag());
//  this->_link->send(gain);
//
//  return wait_msg_processed(msg_id);
//}
//
// bool Controller::send(core::Modulation& mod) {
//  mod.init();
//
//  while (true) {
//    const auto msg_id = mod.prepare(_props.fpga_ctrl_flag(), _props.cpu_ctrl_flag());
//    this->_link->send(mod);
//    // const auto msg_id = core::logic::pack_header(mod, _props.fpga_ctrl_flag(), _props.cpu_ctrl_flag(), &this->_tx_buf[0], &mod_sent);
//    // constexpr auto size = sizeof(core::GlobalHeader);
//    // this->_link->send(&this->_tx_buf[0], size);
//    if (!wait_msg_processed(msg_id)) return false;
//    if (mod.is_finished) return true;
//  }
//}
//
// bool Controller::send(core::Gain& gain, core::Modulation& mod) {
//  size_t mod_sent = 0;
//  mod.build();
//
//  this->_props._output_enable = true;
//  this->_props._op_mode = core::OP_MODE_NORMAL;
//  gain.build(this->_geometry);
//
//  // TODO(ME)
//  while (true) {
//    const auto msg_id = core::logic::pack_header(mod, _props.fpga_ctrl_flag(), _props.cpu_ctrl_flag(), &this->_tx_buf[0], &mod_sent);
//    const auto size = core::logic::pack_body(gain, &this->_tx_buf[0]);
//    this->_link->send(&this->_tx_buf[0], size);
//    if (!wait_msg_processed(msg_id)) return false;
//    if (mod_sent >= mod.buffer().size()) return true;
//  }
//}
//
// bool Controller::send(const core::PointSequence& seq, core::Modulation& mod) {
//  size_t mod_sent = 0;
//  size_t seq_sent = 0;
//  mod.build();
//
//  this->_props._output_enable = true;
//  this->_props._op_mode = core::OP_MODE_SEQ;
//  this->_props._seq_mode = core::SEQ_MODE_POINT;
//
//  while (true) {
//    const auto msg_id = core::logic::pack_header(mod, _props.fpga_ctrl_flag(), _props.cpu_ctrl_flag(), &this->_tx_buf[0], &mod_sent);
//    const auto size = core::logic::pack_body(seq, this->_geometry, &this->_tx_buf[0], &seq_sent);
//    this->_link->send(&this->_tx_buf[0], size);
//    if (!wait_msg_processed(msg_id)) return false;
//    if (seq_sent == seq.control_points().size() && mod_sent == mod.buffer().size()) return true;
//  }
//}
//
// bool Controller::send(const core::GainSequence& seq, core::Modulation& mod) {
//  size_t mod_sent = 0;
//  size_t seq_sent = 0;
//  mod.build();
//
//  for (auto&& g : seq.gains()) g->build(this->_geometry);
//
//  this->_props._output_enable = true;
//  this->_props._op_mode = core::OP_MODE_SEQ;
//  this->_props._seq_mode = core::SEQ_MODE_GAIN;
//
//  while (true) {
//    const auto msg_id = core::logic::pack_header(mod, _props.fpga_ctrl_flag(), _props.cpu_ctrl_flag(), &this->_tx_buf[0], &mod_sent);
//    const auto size = core::logic::pack_body(seq, this->_geometry, &this->_tx_buf[0], &seq_sent);
//    this->_link->send(&this->_tx_buf[0], size);
//    if (!wait_msg_processed(msg_id)) return false;
//    if (seq_sent == seq.gains().size() + 1 && mod_sent == mod.buffer().size()) return true;
//  }
//}

// std::vector<core::DelayOffset>& Controller::delay_offset() { return this->_delay_offset; }
//
// bool Controller::set_delay_offset() { return this->send_delay_offset(); }
//
// bool Controller::send_delay_offset() const {
//   const uint8_t msg_id = core::logic::get_id();
//   core::logic::pack_header(msg_id, _props.fpga_ctrl_flag(), _props.cpu_ctrl_flag() | core::DELAY_OFFSET, &this->_tx_buf[0]);
//   const auto size = core::logic::pack_delay_offset_body(this->_delay_offset, &this->_tx_buf[0]);
//   this->_link->send(&this->_tx_buf[0], size);
//   return wait_msg_processed(msg_id);
// }

std::vector<FirmwareInfo> Controller::firmware_info_list() {
  auto concat_byte = [](const uint8_t high, const uint16_t low) { return static_cast<uint16_t>(static_cast<uint16_t>(high) << 8 | low); };

  const auto num_devices = this->_geometry.num_devices();
  const auto check_ack = this->_check_ack;
  this->_check_ack = true;

  // For backward compatibility before 1.9
  constexpr uint8_t READ_CPU_VER_LSB = 0x02;
  constexpr uint8_t READ_CPU_VER_MSB = 0x03;
  constexpr uint8_t READ_FPGA_VER_LSB = 0x04;
  constexpr uint8_t READ_FPGA_VER_MSB = 0x05;
  auto send_command = [&](const uint8_t msg_id, const uint8_t cmd) {
    core::CommonHeader common_header;
    common_header.init();
    common_header.pack(_geometry, _tx_buf, _props.fpga_ctrl_flag(), _props.cpu_ctrl_flag());
    _tx_buf.header()[2] = cmd;
    _link->send(_tx_buf);
    return wait_msg_processed(msg_id);
  };

  std::vector<uint16_t> cpu_versions(num_devices);
  if (send_command(core::MSG_RD_CPU_V_LSB, READ_CPU_VER_LSB))
    for (size_t i = 0; i < num_devices; i++) cpu_versions[i] = this->_rx_buf[i].ack;
  else
    for (size_t i = 0; i < num_devices; i++) cpu_versions[i] = 0x1;

  if (send_command(core::MSG_RD_CPU_V_MSB, READ_CPU_VER_MSB))
    for (size_t i = 0; i < num_devices; i++) cpu_versions[i] = concat_byte(this->_rx_buf[i].ack, cpu_versions[i]);
  else
    for (size_t i = 0; i < num_devices; i++) cpu_versions[i] = concat_byte(0x1, cpu_versions[i]);

  std::vector<uint16_t> fpga_versions(num_devices);
  if (send_command(core::MSG_RD_FPGA_V_LSB, READ_FPGA_VER_LSB))
    for (size_t i = 0; i < num_devices; i++) fpga_versions[i] = this->_rx_buf[i].ack;
  else
    for (size_t i = 0; i < num_devices; i++) fpga_versions[i] = 0x1;

  if (send_command(core::MSG_RD_FPGA_V_MSB, READ_FPGA_VER_MSB))
    for (size_t i = 0; i < num_devices; i++) fpga_versions[i] = concat_byte(this->_rx_buf[i].ack, fpga_versions[i]);
  else
    for (size_t i = 0; i < num_devices; i++) fpga_versions[i] = concat_byte(0x1, fpga_versions[i]);

  this->_check_ack = check_ack;

  std::vector<FirmwareInfo> infos;
  for (size_t i = 0; i < num_devices; i++) infos.emplace_back(FirmwareInfo(static_cast<uint16_t>(i), cpu_versions[i], fpga_versions[i]));
  return infos;
}

std::unique_ptr<Controller::STMController> Controller::stm() {
  struct Impl : STMController {
    Impl(std::unique_ptr<STMTimerCallback> callback, Controller* p_cnt) : STMController(p_cnt, std::move(callback)) {}
  };
  return std::make_unique<Impl>(std::make_unique<STMTimerCallback>(std::move(this->_link)), this);
}

void Controller::STMController::add_gain(core::Gain& gain) const {
  gain.init();

  auto build_buf = core::TxDatagram(this->_p_cnt->_geometry.num_devices());
  gain.pack(this->_p_cnt->geometry(), build_buf, this->_p_cnt->_props.fpga_ctrl_flag(), this->_p_cnt->_props.cpu_ctrl_flag());

  this->_handler->add(std::move(build_buf));
}

void Controller::STMController::start(const double freq) {
  if (this->_handler == nullptr) throw core::exception::STMError("STM has been already started");

  const auto len = this->_handler->_txs.size();
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

void Controller::STMTimerCallback::add(core::TxDatagram tx) { this->_txs.emplace_back(std::move(tx)); }
void Controller::STMTimerCallback::clear() {
  this->_txs.clear();
  this->_idx = 0;
}

void Controller::STMTimerCallback::callback() {
  if (auto expected = false; _lock.compare_exchange_weak(expected, true)) {
    this->_link->send(this->_txs[this->_idx]);
    this->_idx = (this->_idx + 1) % this->_txs.size();
    _lock.store(false, std::memory_order_release);
  }
}

}  // namespace autd
