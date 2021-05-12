// File: controller.cpp
// Project: lib
// Created Date: 05/11/2020
// Author: Shun Suzuki
// -----
// Last Modified: 13/05/2021
// Modified By: Shun Suzuki (suzuki@hapis.k.u-tokyo.ac.jp)
// -----
// Copyright (c) 2021 Hapis Lab. All rights reserved.
//

#include "controller.hpp"

#include <condition_variable>
#include <iostream>
#include <queue>
#include <vector>

#include "core/ec_config.hpp"
#include "core/logic.hpp"
#include "primitive_gain.hpp"

namespace autd {

bool Controller::is_open() const { return this->_link != nullptr && this->_link->is_open(); }

core::GeometryPtr Controller::geometry() const noexcept { return this->_geometry; }

bool& Controller::silent_mode() noexcept { return this->_silent_mode; }

Result<bool, std::string> Controller::OpenWith(const core::LinkPtr link) {
  if (is_open())
    if (auto close_res = this->Close(); close_res.is_err()) return close_res;

  this->_tx_buf = std::make_unique<uint8_t[]>(this->_geometry->num_devices() * core::EC_OUTPUT_FRAME_SIZE);
  this->_rx_buf = std::make_unique<uint8_t[]>(this->_geometry->num_devices() * core::EC_INPUT_FRAME_SIZE);

  this->_link = link;
  this->_stm = std::make_shared<STMController>(this->_link, this->_geometry, &this->_silent_mode);
  return this->_link->Open();
}

Result<bool, std::string> Controller::Synchronize(const core::Configuration config) {
  this->_config = config;
  uint8_t msg_id = 0;
  core::Logic::PackHeader(core::COMMAND::INIT_FPGA_REF_CLOCK, false, false, &_tx_buf[0], &msg_id);
  size_t size = 0;
  const auto num_devices = this->_geometry->num_devices();
  if (auto res = core::Logic::PackSyncBody(config, num_devices, &_tx_buf[0], &size); res.is_err()) return res;
  std::cout << (int)sizeof(core::RxGlobalHeader) << ", ";

  for (auto i = 0; i < 128; i++) {
    std::cout << (int)_tx_buf[i] << ", ";
  }
  std::cout << "\n";
  for (auto i = 128; i < 626; i++) {
    std::cout << (int)_tx_buf[i] << ", ";
  }
  if (auto res = this->_link->Send(size, &_tx_buf[0]); res.is_err()) return res;
  return WaitMsgProcessed(msg_id, 5000);
}

Result<bool, std::string> Controller::Clear() const { return SendHeader(core::COMMAND::CLEAR); }

Result<bool, std::string> Controller::SendHeader(const core::COMMAND cmd) const {
  const auto send_size = sizeof(core::RxGlobalHeader);
  uint8_t msg_id = 0;
  core::Logic::PackHeader(cmd, false, false, &_tx_buf[0], &msg_id);
  if (auto res = _link->Send(send_size, &_tx_buf[0]); res.is_err()) return res;
  return WaitMsgProcessed(msg_id, 50);
}

Result<bool, std::string> Controller::WaitMsgProcessed(const uint8_t msg_id, const size_t max_trial) const {
  if (_link == nullptr || !_link->is_open()) return Ok(false);

  const auto num_devices = this->_geometry->num_devices();

  const auto buffer_len = num_devices * core::EC_INPUT_FRAME_SIZE;
  for (size_t i = 0; i < max_trial; i++) {
    if (auto res = _link->Read(&_rx_buf[0], buffer_len); res.is_err()) return res;

    size_t processed = 0;
    for (size_t dev = 0; dev < num_devices; dev++)
      if (const uint8_t proc_id = _rx_buf[dev * 2 + 1]; proc_id == msg_id) processed++;

    if (processed == num_devices) return Ok(true);

    auto wait = static_cast<size_t>(
        std::ceil(static_cast<double>(core::EC_TRAFFIC_DELAY) * 1000 / core::EC_DEVICE_PER_FRAME * static_cast<double>(num_devices)));
    std::this_thread::sleep_for(std::chrono::milliseconds(wait));
  }

  return Ok(false);
}

Result<bool, std::string> Controller::Close() {
  Result<bool, std::string> res = Ok(true);
  res = this->_stm->Finish();
  if (res.is_err()) return res;

  res = this->Stop();
  if (res.is_err()) return res;

  res = this->Clear();
  if (res.is_err()) return res;

  res = this->_link->Close();
  this->_link = nullptr;
  this->_tx_buf = nullptr;
  this->_rx_buf = nullptr;

  return res;
}

Result<bool, std::string> Controller::Stop() const {
  const auto null_gain = gain::NullGain::Create();
  return this->Send(null_gain, nullptr, false);
}

Result<bool, std::string> Controller::Send(const core::GainPtr gain, const core::ModulationPtr mod, const bool wait_for_sent) const {
  Result<bool, std::string> res = Ok(true);

  res = this->_stm->Stop();
  if (res.is_err()) return res;

  if (mod != nullptr) {
    res = mod->Build(this->_config);
    if (res.is_err()) return res;
  }

  uint8_t msg_id = 0;
  core::Logic::PackHeader(mod, this->_silent_mode, this->_seq_mode, &this->_tx_buf[0], &msg_id);

  if (gain != nullptr) {
    res = gain->Build(this->_geometry);
    if (res.is_err()) return res;
  }

  size_t size = 0;
  core::Logic::PackBody(gain, &this->_tx_buf[0], &size);
  res = this->_link->Send(size, &this->_tx_buf[0]);
  if (res.is_err()) return res;

  if (!wait_for_sent) return res;

  return WaitMsgProcessed(msg_id);
}

Result<std::vector<FirmwareInfo>, std::string> Controller::firmware_info_list() const {
  auto concat_byte = [](const uint8_t high, const uint16_t low) { return static_cast<uint16_t>(static_cast<uint16_t>(high) << 8 | low); };

  Result<bool, std::string> res = Ok(true);

  const auto num_devices = this->_geometry->num_devices();
  std::vector<uint16_t> cpu_versions(num_devices);
  res = SendHeader(core::COMMAND::READ_CPU_VER_LSB);
  if (res.is_err()) return Err(res.unwrap_err());
  for (size_t i = 0; i < num_devices; i++) cpu_versions[i] = _rx_buf[2 * i];
  res = SendHeader(core::COMMAND::READ_CPU_VER_MSB);
  if (res.is_err()) return Err(res.unwrap_err());
  for (size_t i = 0; i < num_devices; i++) cpu_versions[i] = concat_byte(_rx_buf[2 * i], cpu_versions[i]);

  std::vector<uint16_t> fpga_versions(num_devices);
  res = SendHeader(core::COMMAND::READ_FPGA_VER_LSB);
  if (res.is_err()) return Err(res.unwrap_err());
  for (size_t i = 0; i < num_devices; i++) fpga_versions[i] = _rx_buf[2 * i];
  res = SendHeader(core::COMMAND::READ_FPGA_VER_MSB);
  if (res.is_err()) return Err(res.unwrap_err());
  for (size_t i = 0; i < num_devices; i++) fpga_versions[i] = concat_byte(_rx_buf[2 * i], fpga_versions[i]);

  std::vector<FirmwareInfo> infos;
  for (size_t i = 0; i < num_devices; i++) infos.emplace_back(FirmwareInfo(static_cast<uint16_t>(i), cpu_versions[i], fpga_versions[i]));
  return Ok(std::move(infos));
}

std::shared_ptr<Controller::STMController> Controller::stm() const { return this->_stm; }

void Controller::STMController::AddGain(const core::GainPtr gain) { _gains.emplace_back(gain); }
void Controller::STMController::AddGains(const std::vector<core::GainPtr>& gains) {
  for (const auto& g : gains) this->AddGain(g);
}

[[nodiscard]] Result<bool, std::string> Controller::STMController::Start(const double freq) {
  auto len = this->_gains.size();
  auto interval_us = static_cast<uint32_t>(1000000. / static_cast<double>(freq) / static_cast<double>(len));
  this->_timer.SetInterval(interval_us);

  const auto current_size = this->_bodies.size();
  this->_bodies.resize(len);
  this->_body_sizes.resize(len);

  for (auto i = current_size; i < len; i++) {
    auto& g = this->_gains[i];
    if (auto res = g->Build(this->_geometry); res.is_err()) return res;

    size_t body_size = 0;
    uint8_t msg_id = 0;
    this->_bodies[i] = new uint8_t[body_size];
    core::Logic::PackHeader(nullptr, *this->_silent_mode, false, this->_bodies[i], &msg_id);
    core::Logic::PackBody(g, this->_bodies[i], &body_size);
    this->_body_sizes[i] = body_size;
  }

  size_t idx = 0;
  return this->_timer.Start([this, idx, len]() mutable {
    if (auto expected = false; this->_lock.compare_exchange_weak(expected, true)) {
      const auto body_size = this->_body_sizes[idx];
      if (const auto res = this->_link->Send(body_size, this->_bodies[idx]); res.is_err()) return;
      idx = (idx + 1) % len;
      this->_lock.store(false, std::memory_order_release);
    }
  });
}

[[nodiscard]] Result<bool, std::string> Controller::STMController::Stop() { return this->_timer.Stop(); }

[[nodiscard]] Result<bool, std::string> Controller::STMController::Finish() {
  if (auto res = this->Stop(); res.is_err()) return res;

  std::vector<core::GainPtr>().swap(this->_gains);
  for (auto* p : this->_bodies) delete[] p;
  std::vector<uint8_t*>().swap(this->_bodies);
  std::vector<size_t>().swap(this->_body_sizes);

  return Ok(true);
}
}  // namespace autd
