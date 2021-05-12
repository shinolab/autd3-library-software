// File: controller.cpp
// Project: lib
// Created Date: 05/11/2020
// Author: Shun Suzuki
// -----
// Last Modified: 12/05/2021
// Modified By: Shun Suzuki (suzuki@hapis.k.u-tokyo.ac.jp)
// -----
// Copyright (c) 2021 Hapis Lab. All rights reserved.
//

#include "controller.hpp"

#include <condition_variable>
#include <mutex>
#include <queue>
#include <thread>
#include <utility>
#include <vector>

#include "core/logic.hpp"
#include "primitive_gain.hpp"

namespace autd {

using std::move;
using std::shared_ptr;
using std::thread, std::queue;
using std::vector, std::condition_variable, std::unique_lock, std::mutex;

bool Controller::is_open() const { return this->_link != nullptr && this->_link->is_open(); }

core::GeometryPtr Controller::geometry() const noexcept { return this->_geometry; }

bool& Controller::silent_mode() noexcept { return this->_silent_mode; }

Result<bool, std::string> Controller::OpenWith(core::LinkPtr link) {
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
  return core::Logic::Synchronize(this->_link, this->_geometry->num_devices(), &_tx_buf[0], &_rx_buf[0], _config);
}

Result<bool, std::string> Controller::Clear() const {
  return core::Logic::Clear(this->_link, this->_geometry->num_devices(), &_tx_buf[0], &_rx_buf[0]);
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

Result<bool, std::string> Controller::Stop() {
  const auto null_gain = gain::NullGain::Create();
  return this->Send(null_gain, nullptr, false);
}

Result<bool, std::string> Controller::Send(const core::GainPtr gain, const core::ModulationPtr mod, const bool wait_for_sent) {
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

  return core::Logic::WaitMsgProcessed(this->_link, msg_id, this->_geometry->num_devices(), &this->_tx_buf[0]);
}

Result<std::vector<FirmwareInfo>, std::string> Controller::firmware_info_list() const {
  return core::Logic::firmware_info_list(this->_link, this->_geometry->num_devices(), &this->_tx_buf[0], &this->_rx_buf[0]);
}

std::shared_ptr<Controller::STMController> Controller::stm() const { return this->_stm; }

void Controller::STMController::AddGain(core::GainPtr gain) { _gains.emplace_back(gain); }
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

  vector<core::GainPtr>().swap(this->_gains);
  for (auto* p : this->_bodies) delete[] p;
  vector<uint8_t*>().swap(this->_bodies);
  vector<size_t>().swap(this->_body_sizes);

  return Ok(true);
}
}  // namespace autd
