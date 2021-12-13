// File: controller.cpp
// Project: lib
// Created Date: 05/11/2020
// Author: Shun Suzuki
// -----
// Last Modified: 13/12/2021
// Modified By: Shun Suzuki (suzuki@hapis.k.u-tokyo.ac.jp)
// -----
// Copyright (c) 2021 Hapis Lab. All rights reserved.
//

#include "autd3/controller.hpp"

#include <condition_variable>
#include <vector>

#include "autd3/core/ec_config.hpp"
#include "autd3/core/interface.hpp"
#include "autd3/core/logic.hpp"
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

const std::vector<core::FPGAInfo>& Controller::fpga_info() {
  this->_link->receive(this->_rx_buf);
  for (size_t i = 0; i < this->_geometry.num_devices(); i++) this->_fpga_infos[i].set(_rx_buf[i]);
  return _fpga_infos;
}

bool Controller::update_ctrl_flag() {
  core::CommonHeader header(core::OUTPUT_ENABLE | core::OUTPUT_BALANCE | core::SILENT | core::READS_FPGA_INFO | core::FORCE_FAN);
  return send(header);
}

void Controller::open(core::LinkPtr link) {
  this->close();

  this->_tx_buf = core::TxDatagram(this->_geometry.num_devices());
  this->_rx_buf = core::RxDatagram(this->_geometry.num_devices());

  this->_fpga_infos.resize(this->_geometry.num_transducers());

  link->open();
  this->_link = std::move(link);
}

bool Controller::clear() {
  core::SpecialMessageIdHeader header(core::MSG_CLEAR, 0xFF);
  return send(header);
}

bool Controller::wait_msg_processed(const uint8_t msg_id, const size_t max_trial) {
  if (!this->_check_ack) return true;
  const auto num_devices = this->_geometry.num_devices();
  for (size_t i = 0; i < max_trial; i++) {
    this->_link->receive(this->_rx_buf);
    if (is_msg_processed(num_devices, msg_id, _rx_buf)) return true;
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
  core::NullBody body;
  return this->send(header, body);
}

bool Controller::send(core::IDatagramBody& body) {
  core::CommonHeader header(core::OUTPUT_ENABLE | core::OUTPUT_BALANCE | core::SILENT | core::READS_FPGA_INFO | core::FORCE_FAN);
  return this->send(header, body);
}

bool Controller::send(core::IDatagramBody& body, core::IDatagramHeader& header) { return this->send(header, body); }

bool Controller::send(core::IDatagramHeader& header, core::IDatagramBody& body) {
  header.init();
  body.init();

  while (true) {
    const auto msg_id = header.pack(_tx_buf, _props.fpga_ctrl_flag(), _props.cpu_ctrl_flag());
    body.pack(this->_geometry, _tx_buf);
    this->_link->send(this->_tx_buf);
    if (!wait_msg_processed(msg_id)) return false;
    if (header.is_finished() && body.is_finished()) return true;
  }
}

std::vector<FirmwareInfo> Controller::firmware_info_list() {
  const auto check_ack = this->_check_ack;
  this->_check_ack = true;

  // For backward compatibility before 1.9
  constexpr uint8_t READ_CPU_VER_LSB = 0x02;
  constexpr uint8_t READ_CPU_VER_MSB = 0x03;
  constexpr uint8_t READ_FPGA_VER_LSB = 0x04;
  constexpr uint8_t READ_FPGA_VER_MSB = 0x05;
  auto send_command = [&](const uint8_t msg_id, const uint8_t cmd) {
    core::SpecialMessageIdHeader common_header(msg_id,
                                               core::OUTPUT_ENABLE | core::OUTPUT_BALANCE | core::SILENT | core::READS_FPGA_INFO | core::FORCE_FAN);
    core::NullBody body;

    common_header.init();
    body.init();

    common_header.pack(_tx_buf, _props.fpga_ctrl_flag(), _props.cpu_ctrl_flag());
    body.pack(this->_geometry, _tx_buf);
    _tx_buf.header()[2] = cmd;
    _link->send(_tx_buf);
    return wait_msg_processed(msg_id);
  };

  std::vector<uint16_t> cpu_versions_lsb;
  if (send_command(core::MSG_RD_CPU_V_LSB, READ_CPU_VER_LSB))
    for (auto& [ack, _] : this->_rx_buf) cpu_versions_lsb.emplace_back(static_cast<uint16_t>(ack));
  else
    for (auto& _ : this->_rx_buf) cpu_versions_lsb.emplace_back(0x0000);

  std::vector<uint16_t> cpu_versions_msb;
  if (send_command(core::MSG_RD_CPU_V_MSB, READ_CPU_VER_MSB))
    for (auto& [ack, _] : this->_rx_buf) cpu_versions_msb.emplace_back(static_cast<uint16_t>(ack) << 8);
  else
    for (auto& _ : this->_rx_buf) cpu_versions_msb.emplace_back(0x0000);

  std::vector<uint16_t> fpga_versions_lsb;
  if (send_command(core::MSG_RD_FPGA_V_LSB, READ_FPGA_VER_LSB))
    for (auto& [ack, _] : this->_rx_buf) fpga_versions_lsb.emplace_back(static_cast<uint16_t>(ack));
  else
    for (auto& _ : this->_rx_buf) fpga_versions_lsb.emplace_back(0x0000);

  std::vector<uint16_t> fpga_versions_msb;
  if (send_command(core::MSG_RD_FPGA_V_MSB, READ_FPGA_VER_MSB))
    for (auto& [ack, _] : this->_rx_buf) fpga_versions_msb.emplace_back(static_cast<uint16_t>(ack) << 8);
  else
    for (auto& _ : this->_rx_buf) fpga_versions_msb.emplace_back(0x0000);

  this->_check_ack = check_ack;

  std::vector<FirmwareInfo> infos;
  for (size_t i = 0; i < this->_geometry.num_devices(); i++)
    infos.emplace_back(
        FirmwareInfo(static_cast<uint16_t>(i), cpu_versions_msb[i] | cpu_versions_lsb[i], fpga_versions_msb[i] | fpga_versions_lsb[i]));
  return infos;
}

Controller::STMController Controller::stm() { return STMController{this, std::make_unique<STMTimerCallback>(std::move(this->_link))}; }

void Controller::STMController::add(core::Gain& gain) const {
  core::TxDatagram build_buf(this->_p_cnt->_geometry.num_devices());
  core::CommonHeader header(core::OUTPUT_ENABLE | core::OUTPUT_BALANCE | core::SILENT | core::READS_FPGA_INFO | core::FORCE_FAN);

  header.init();
  gain.init();

  header.pack(build_buf, this->_p_cnt->_props.fpga_ctrl_flag(), this->_p_cnt->_props.cpu_ctrl_flag());
  gain.pack(this->_p_cnt->geometry(), build_buf);

  this->_handler->add(std::move(build_buf));
}

void Controller::STMController::add(core::Gain&& gain) const {
  core::TxDatagram build_buf(this->_p_cnt->_geometry.num_devices());
  core::CommonHeader header(core::OUTPUT_ENABLE | core::OUTPUT_BALANCE | core::SILENT | core::READS_FPGA_INFO | core::FORCE_FAN);

  header.init();
  gain.init();

  header.pack(build_buf, this->_p_cnt->_props.fpga_ctrl_flag(), this->_p_cnt->_props.cpu_ctrl_flag());
  gain.pack(this->_p_cnt->geometry(), build_buf);

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
  if (_p_cnt == nullptr) return;
  this->stop();
  this->_handler->clear();
  this->_p_cnt->_link = std::move(this->_handler->_link);
  this->_p_cnt = nullptr;
  this->_handler = nullptr;
  this->_timer = nullptr;
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
