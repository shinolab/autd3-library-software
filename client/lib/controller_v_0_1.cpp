// File: controller.cpp
// Project: lib
// Created Date: 13/05/2016
// Author: Seki Inoue
// -----
// Last Modified: 12/10/2020
// Modified By: Shun Suzuki (suzuki@hapis.k.u-tokyo.ac.jp)
// -----
// Copyright (c) 2016-2020 Hapis Lab. All rights reserved.
//

#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <cstring>
#include <iostream>
#include <mutex>
#include <queue>
#include <string>
#include <thread>
#include <vector>

#include "controller.hpp"
#include "controller_impl.hpp"
#include "ec_config.hpp"
#include "emulator_link.hpp"
#include "firmware_version.hpp"
#include "geometry.hpp"
#include "link.hpp"
#include "privdef.hpp"
#include "sequence.hpp"
#include "timer.hpp"

namespace autd {
namespace _internal {

AUTDControllerV_0_1::AUTDControllerV_0_1() : AUTDController() {}

AUTDControllerV_0_1::~AUTDControllerV_0_1() {}

bool AUTDControllerV_0_1::Calibrate() {
  std::cerr << "The function 'Calibrate' does not work in this version." << std::endl;
  return false;
}

bool AUTDControllerV_0_1::Clear() {
  std::cerr << "The function 'Clear' does not work in this version." << std::endl;
  return false;
}

void AUTDControllerV_0_1::Close() {
  if (this->is_open()) {
    this->FinishSTModulation();
    this->Flush();
    this->Stop();
    this->CloseLink();
    this->_build_cond.notify_all();
    if (std::this_thread::get_id() != this->_build_thr.get_id() && this->_build_thr.joinable()) this->_build_thr.join();
    this->_send_cond.notify_all();
    if (std::this_thread::get_id() != this->_send_thr.get_id() && this->_send_thr.joinable()) this->_send_thr.join();
    this->_link = nullptr;
  }
}

void AUTDControllerV_0_1::Stop() {
  auto nullgain = autd::gain::NullGain::Create();
  this->AppendGainSync(nullgain, true);
}

void AUTDControllerV_0_1::AppendGain(GainPtr gain) {
  this->_p_stm_timer->Stop();
  gain->SetGeometry(this->_geometry);
  {
    std::unique_lock<std::mutex> lk(_build_mtx);
    _build_q.push(gain);
  }
  _build_cond.notify_all();
}

void AUTDControllerV_0_1::AppendGainSync(GainPtr gain, bool _wait_for_send) {
  this->_p_stm_timer->Stop();
  try {
    gain->SetGeometry(this->_geometry);
    gain->Build();

    size_t body_size = 0;
    uint8_t msg_id = 0;
    auto body = this->MakeBody(gain, nullptr, &body_size, &msg_id);
    if (this->is_open()) this->SendData(body_size, move(body));
  } catch (const int errnum) {
    this->CloseLink();
    std::cerr << errnum << "Link closed." << std::endl;
  }
}

void AUTDControllerV_0_1::AppendModulation(ModulationPtr mod) {
  {
    std::unique_lock<std::mutex> lk(_send_mtx);
    _send_mod_q.push(mod);
  }
  _send_cond.notify_all();
}

void AUTDControllerV_0_1::AppendModulationSync(ModulationPtr mod) {
  try {
    if (this->is_open()) {
      while (mod->buffer.size() > this->mod_sent(mod)) {
        size_t body_size = 0;
        uint8_t msg_id = 0;
        auto body = this->MakeBody(nullptr, mod, &body_size, &msg_id);
        this->SendData(body_size, move(body));
      }
      this->mod_sent(mod) = 0;
    }
  } catch (const int errnum) {
    this->Close();
    std::cerr << errnum << "Link closed." << std::endl;
  }
}

void AUTDControllerV_0_1::AppendSTMGain(GainPtr gain) { _stm_gains.push_back(gain); }
void AUTDControllerV_0_1::AppendSTMGain(const std::vector<GainPtr> &gain_list) {
  for (auto g : gain_list) {
    this->AppendSTMGain(g);
  }
}

void AUTDControllerV_0_1::StartSTModulation(double freq) {
  auto len = this->_stm_gains.size();
  auto itvl_us = static_cast<int>(1000000. / freq / len);
  this->_p_stm_timer->SetInterval(itvl_us);

  auto current_size = this->_stm_bodies.size();
  this->_stm_bodies.resize(len);
  this->_stm_body_sizes.resize(len);

  for (size_t i = current_size; i < len; i++) {
    auto g = this->_stm_gains[i];
    g->SetGeometry(this->_geometry);
    g->Build();

    size_t body_size = 0;
    uint8_t msg_id = 0;
    auto body = this->MakeBody(g, nullptr, &body_size, &msg_id);
    uint8_t *b = new uint8_t[body_size];
    std::memcpy(b, body.get(), body_size);
    this->_stm_bodies[i] = b;
    this->_stm_body_sizes[i] = body_size;
  }

  size_t idx = 0;
  this->_p_stm_timer->Start([this, idx, len]() mutable {
    auto body_size = this->_stm_body_sizes[idx];
    auto body_copy = std::make_unique<uint8_t[]>(body_size);
    uint8_t *p = this->_stm_bodies[idx];
    std::memcpy(body_copy.get(), p, body_size);
    if (this->is_open()) this->SendData(body_size, std::move(body_copy));
    idx = (idx + 1) % len;
  });
}

void AUTDControllerV_0_1::StopSTModulation() {
  this->_p_stm_timer->Stop();
  this->Stop();
}

void AUTDControllerV_0_1::FinishSTModulation() {
  this->StopSTModulation();
  std::vector<GainPtr>().swap(this->_stm_gains);
  for (uint8_t *p : this->_stm_bodies) {
    delete[] p;
  }
  std::vector<uint8_t *>().swap(this->_stm_bodies);
  std::vector<size_t>().swap(this->_stm_body_sizes);
}

void AUTDControllerV_0_1::AppendSequence(SequencePtr seq) { throw "Sequence is not implemented yet and is available since v0.6."; }

void AUTDControllerV_0_1::Flush() {
  std::unique_lock<std::mutex> lk0(_send_mtx);
  std::unique_lock<std::mutex> lk1(_build_mtx);
  std::queue<GainPtr>().swap(_build_q);
  std::queue<GainPtr>().swap(_send_gain_q);
  std::queue<ModulationPtr>().swap(_send_mod_q);
}

FirmwareInfoList AUTDControllerV_0_1::firmware_info_list() {
  std::cerr << "The function 'firmware_info_list' does not work in this version." << std::endl;
  auto size = this->_geometry->numDevices();
  FirmwareInfoList res;
  for (uint16_t i = 0; i < static_cast<uint16_t>(size); i++) {
    auto info = AUTDController::FirmwareInfoCreate(i, 0, 0);
    res.push_back(info);
  }
  return res;
}

void AUTDControllerV_0_1::LateralModulationAT(Vector3 point, Vector3 dir, double lm_amp, double lm_freq) {
  auto p1 = point + lm_amp * dir;
  auto p2 = point - lm_amp * dir;
  this->FinishSTModulation();
  this->AppendSTMGain(autd::gain::FocalPointGain::Create(p1));
  this->AppendSTMGain(autd::gain::FocalPointGain::Create(p2));
  this->StartSTModulation(lm_freq);
}

void AUTDControllerV_0_1::InitPipeline() {
  this->_build_thr = std::thread([&] {
    while (this->is_open()) {
      GainPtr gain = nullptr;
      {
        std::unique_lock<std::mutex> lk(_build_mtx);

        _build_cond.wait(lk, [&] { return _build_q.size() || !this->is_open(); });

        if (_build_q.size() > 0) {
          gain = _build_q.front();
          _build_q.pop();
        }
      }

      if (gain != nullptr) {
        gain->Build();
        {
          std::unique_lock<std::mutex> lk(_send_mtx);
          _send_gain_q.push(gain);
          _send_cond.notify_all();
        }
      }
    }
  });

  this->_send_thr = std::thread([&] {
    try {
      while (this->is_open()) {
        GainPtr gain = nullptr;
        ModulationPtr mod = nullptr;

        {
          std::unique_lock<std::mutex> lk(_send_mtx);
          _send_cond.wait(lk, [&] { return _send_gain_q.size() || _send_mod_q.size() || !this->is_open(); });
          if (_send_gain_q.size() > 0) gain = _send_gain_q.front();
          if (_send_mod_q.size() > 0) mod = _send_mod_q.front();
        }
        size_t body_size = 0;
        uint8_t msg_id = 0;
        auto body = MakeBody(gain, mod, &body_size, &msg_id);
        if (this->is_open()) this->SendData(body_size, move(body));

        std::unique_lock<std::mutex> lk(_send_mtx);
        if (gain != nullptr && _send_gain_q.size() > 0) _send_gain_q.pop();
        if (mod != nullptr && mod->buffer.size() <= this->mod_sent(mod)) {
          this->mod_sent(mod) = 0;
          if (_send_mod_q.size() > 0) _send_mod_q.pop();
        }
      }
    } catch (const int errnum) {
      this->Close();
      std::cerr << errnum << "Link closed." << std::endl;
    }
  });
}

std::unique_ptr<uint8_t[]> AUTDControllerV_0_1::MakeBody(GainPtr gain, ModulationPtr mod, size_t *const size, uint8_t *const send_msg_id) {
  auto num_devices = (gain != nullptr) ? gain->geometry()->numDevices() : 0;

  *size = sizeof(RxGlobalHeaderV_0_1) + sizeof(uint16_t) * NUM_TRANS_IN_UNIT * num_devices;
  auto body = std::make_unique<uint8_t[]>(*size);

  auto *header = reinterpret_cast<RxGlobalHeaderV_0_1 *>(&body[0]);
  *send_msg_id = get_id();
  header->msg_id = *send_msg_id;
  header->control_flags = 0;
  header->mod_size = 0;

  if (this->_silent_mode) header->control_flags |= SILENT;

  if (mod != nullptr) {
    const uint8_t mod_size = std::max(0, std::min(static_cast<int>(mod->buffer.size() - this->mod_sent(mod)), MOD_FRAME_SIZE_V_0_1));
    header->mod_size = mod_size;
    if (this->mod_sent(mod) == 0) header->control_flags |= MOD_BEGIN | LOOP_BEGIN;
    if (this->mod_sent(mod) + mod_size >= mod->buffer.size()) header->control_flags |= LOOP_END;

    std::memcpy(header->mod, &mod->buffer[this->mod_sent(mod)], mod_size);
    this->mod_sent(mod) += mod_size;
  }

  auto *cursor = &body[0] + sizeof(RxGlobalHeaderV_0_1) / sizeof(body[0]);
  if (gain != nullptr) {
    for (int i = 0; i < gain->geometry()->numDevices(); i++) {
      auto deviceId = gain->geometry()->deviceIdForDeviceIdx(i);
      auto byteSize = NUM_TRANS_IN_UNIT * sizeof(uint16_t);
      std::memcpy(cursor, this->gain_data_addr(gain, deviceId), byteSize);
      cursor += byteSize / sizeof(body[0]);
    }
  }
  return body;
}
}  // namespace _internal
}  // namespace autd
