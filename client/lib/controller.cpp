// File: controller.cpp
// Project: lib
// Created Date: 13/05/2016
// Author: Seki Inoue
// -----
// Last Modified: 27/02/2020
// Modified By: Shun Suzuki (suzuki@hapis.k.u-tokyo.ac.jp)
// -----
// Copyright (c) 2016-2020 Hapis Lab. All rights reserved.
//

#include "controller.hpp"

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

#include "geometry.hpp"
#include "link.hpp"
#include "privdef.hpp"
#if WIN32
#include "ethercat_link.hpp"
#endif
#include "soem_link.hpp"
#include "timer.hpp"

namespace autd {

EtherCATAdapters Controller::EnumerateAdapters(int *const size) {
  auto adapters = libsoem::EtherCATAdapterInfo::EnumerateAdapters();
  *size = static_cast<int>(adapters.size());
#if DLL_FOR_CAPI
  EtherCATAdapters res = new EtherCATAdapter[*size];
  int i = 0;
#else
  EtherCATAdapters res;
#endif
  for (auto adapter : libsoem::EtherCATAdapterInfo::EnumerateAdapters()) {
    EtherCATAdapter p;
    p.first = adapter.desc;
    p.second = adapter.name;
#if DLL_FOR_CAPI
    res[i++] = p;
#else
    res.push_back(p);
#endif
  }
  return res;
}

class AUTDController : public Controller {
 public:
  AUTDController();
  ~AUTDController();

  bool is_open() final;
  GeometryPtr geometry() noexcept final;
  bool silent_mode() noexcept final;
  size_t remainingInBuffer() final;

  void Open(LinkType type, std::string location = "") final;
  void SetSilentMode(bool silent) noexcept final;
  void CalibrateModulation() final;
  void Close() final;

  void Stop() final;
  void AppendGain(const GainPtr gain) final;
  void AppendGainSync(const GainPtr gain) final;
  void AppendModulation(const ModulationPtr mod) final;
  void AppendModulationSync(const ModulationPtr mod) final;
  void AppendSTMGain(GainPtr gain) final;
  void AppendSTMGain(const std::vector<GainPtr> &gain_list) final;
  void StartSTModulation(double freq) final;
  void StopSTModulation() final;
  void FinishSTModulation() final;
  void Flush() final;

  void LateralModulationAT(Vector3 point, Vector3 dir, double lm_amp = 2.5, double lm_freq = 100) final;
  void AppendLateralGain(GainPtr gain) final;
  void AppendLateralGain(const std::vector<GainPtr> &gain_list) final;
  void StartLateralModulation(double freq) final;
  void FinishLateralModulation() final;
  void ResetLateralGain() final;

 private:
  GeometryPtr _geometry;
  std::shared_ptr<internal::Link> _link;
  std::queue<GainPtr> _build_q;
  std::queue<GainPtr> _send_gain_q;
  std::queue<ModulationPtr> _send_mod_q;

  std::vector<GainPtr> _stm_gains;
  std::vector<uint8_t *> _stm_bodies;
  std::vector<size_t> _stm_body__sizes;
  std::unique_ptr<Timer> _p_stm_timer;

  std::thread _build_thr;
  std::thread _send_thr;
  std::condition_variable _build_cond;
  std::condition_variable _send_cond;
  std::mutex _build_mtx;
  std::mutex _send_mtx;

  bool _silent_mode = true;

  void InitPipeline();
  std::unique_ptr<uint8_t[]> MakeBody(GainPtr gain, ModulationPtr mod, size_t *size);

  static uint8_t get_id() {
    static std::atomic<uint8_t> id{0};

    id.fetch_add(0x01);
    uint8_t expected = 0xf0;
    id.compare_exchange_weak(expected, 0x01);

    return id.load();
  }
};

AUTDController::AUTDController() {
  this->_geometry = Geometry::Create();
  this->_silent_mode = true;
  this->_p_stm_timer = std::make_unique<Timer>();
}

AUTDController::~AUTDController() {
  if (std::this_thread::get_id() != this->_build_thr.get_id() && this->_build_thr.joinable()) this->_build_thr.join();
  if (std::this_thread::get_id() != this->_send_thr.get_id() && this->_send_thr.joinable()) this->_send_thr.join();
}

bool AUTDController::is_open() { return this->_link.get() && this->_link->is_open(); }
GeometryPtr AUTDController::geometry() noexcept { return this->_geometry; }
bool AUTDController::silent_mode() noexcept { return this->_silent_mode; }
size_t AUTDController::remainingInBuffer() { return this->_send_gain_q.size() + this->_send_mod_q.size() + this->_build_q.size(); }

void AUTDController::Open(LinkType type, std::string location) {
  this->Close();

  switch (type) {
#if WIN32
    case LinkType::ETHERCAT:
    case LinkType::TwinCAT: {
      if (location == "" || location.find("localhost") == 0 || location.find("0.0.0.0") == 0 || location.find("127.0.0.1") == 0) {
        this->_link = std::make_shared<internal::LocalEthercatLink>();
      } else {
        this->_link = std::make_shared<internal::EthercatLink>();
      }
      this->_link->Open(location);
      break;
    }
#endif
    case LinkType::SOEM: {
      this->_link = std::make_shared<internal::SOEMLink>();
      auto device_num = this->_geometry->numDevices();
      this->_link->Open(location + ":" + std::to_string(device_num));
      break;
    }
    default:
      throw std::runtime_error("This link type is not implemented yet.");
      break;
  }

  if (this->_link->is_open())
    this->InitPipeline();
  else
    this->Close();
}
void AUTDController::SetSilentMode(bool silent) noexcept { this->_silent_mode = silent; }
void AUTDController::CalibrateModulation() { this->_link->CalibrateModulation(); }
void AUTDController::Close() {
  if (this->is_open()) {
    this->FinishSTModulation();
    this->Stop();
    this->_link->Close();
    this->Flush();
    this->_build_cond.notify_all();
    if (std::this_thread::get_id() != this->_build_thr.get_id() && this->_build_thr.joinable()) this->_build_thr.join();
    this->_send_cond.notify_all();
    if (std::this_thread::get_id() != this->_send_thr.get_id() && this->_send_thr.joinable()) this->_send_thr.join();
    this->_link = std::shared_ptr<internal::Link>(nullptr);
  }
}

void AUTDController::Stop() {
  auto nullgain = NullGain::Create();
  this->AppendGainSync(nullgain);
#if DLL_FOR_CAPI
  delete nullgain;
#endif
}
void AUTDController::AppendGain(GainPtr gain) {
  this->_p_stm_timer->Stop();
  {
    gain->SetGeometry(this->_geometry);
    std::unique_lock<std::mutex> lk(_build_mtx);
    _build_q.push(gain);
  }
  _build_cond.notify_all();
}
void AUTDController::AppendGainSync(GainPtr gain) {
  this->_p_stm_timer->Stop();
  try {
    gain->SetGeometry(this->_geometry);
    if (!gain->built()) gain->Build();

    size_t body_size = 0;
    auto body = this->MakeBody(gain, nullptr, &body_size);

    if (this->is_open()) this->_link->Send(body_size, move(body));
  } catch (const int errnum) {
    this->_link->Close();
    std::cerr << errnum << "Link closed." << std::endl;
  }
}
void AUTDController::AppendModulation(ModulationPtr mod) {
  std::unique_lock<std::mutex> lk(_send_mtx);
  _send_mod_q.push(mod);
  _send_cond.notify_all();
}
void AUTDController::AppendModulationSync(ModulationPtr mod) {
  try {
    if (this->is_open()) {
      while (mod->buffer.size() > mod->_sent) {
        size_t body_size = 0;
        auto body = this->MakeBody(nullptr, mod, &body_size);
        this->_link->Send(body_size, move(body));
      }
      mod->_sent = 0;
    }
  } catch (const int errnum) {
    this->Close();
    std::cerr << errnum << "Link closed." << std::endl;
  }
}
void AUTDController::AppendSTMGain(GainPtr gain) { _stm_gains.push_back(gain); }
void AUTDController::AppendSTMGain(const std::vector<GainPtr> &gain_list) {
  for (auto g : gain_list) {
    this->AppendSTMGain(g);
  }
}

void AUTDController::StartSTModulation(double freq) {
  this->_link->SetWaitForProcessMsg(false);

  auto len = this->_stm_gains.size();
  auto itvl_us = static_cast<int>(1000000. / freq / len);
  this->_p_stm_timer->SetInterval(itvl_us);

  auto current_size = this->_stm_bodies.size();
  this->_stm_bodies.resize(len);
  this->_stm_body__sizes.resize(len);

  for (size_t i = current_size; i < len; i++) {
    auto g = this->_stm_gains[i];
    g->SetGeometry(this->_geometry);
    if (!g->built()) g->Build();

    size_t body_size = 0;
    auto body = this->MakeBody(g, nullptr, &body_size);
    uint8_t *b = new uint8_t[body_size];
    std::memcpy(b, body.get(), body_size);
    this->_stm_bodies[i] = b;
    this->_stm_body__sizes[i] = body_size;
  }

  size_t idx = 0;
  this->_p_stm_timer->Start([this, idx, len]() mutable {
    auto body_size = this->_stm_body__sizes[idx];
    auto body_copy = std::make_unique<uint8_t[]>(body_size);
    uint8_t *p = this->_stm_bodies[idx];
    std::memcpy(body_copy.get(), p, body_size);
    if (this->is_open()) this->_link->Send(body_size, std::move(body_copy));
    idx = (idx + 1) % len;
  });
}

void AUTDController::StopSTModulation() {
  this->_p_stm_timer->Stop();
  this->Stop();
}

void AUTDController::FinishSTModulation() {
  this->StopSTModulation();
  std::vector<GainPtr>().swap(this->_stm_gains);
  for (uint8_t *p : this->_stm_bodies) {
    delete[] p;
  }
  std::vector<uint8_t *>().swap(this->_stm_bodies);
  std::vector<size_t>().swap(this->_stm_body__sizes);
  this->_link->SetWaitForProcessMsg(true);
}

void AUTDController::Flush() {
  std::unique_lock<std::mutex> lk0(_send_mtx);
  std::unique_lock<std::mutex> lk1(_build_mtx);
  std::queue<GainPtr>().swap(_build_q);
  std::queue<GainPtr>().swap(_send_gain_q);
  std::queue<ModulationPtr>().swap(_send_mod_q);
}

void AUTDController::LateralModulationAT(Vector3 point, Vector3 dir, double lm_amp, double lm_freq) {
  auto p1 = point + lm_amp * dir;
  auto p2 = point - lm_amp * dir;
  this->FinishSTModulation();
  this->AppendSTMGain(autd::FocalPointGain::Create(p1));
  this->AppendSTMGain(autd::FocalPointGain::Create(p2));
  this->StartSTModulation(lm_freq);
}

void AUTDController::InitPipeline() {
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
        if (!gain->built()) gain->Build();
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
        auto body = MakeBody(gain, mod, &body_size);
        if (this->_link->is_open()) this->_link->Send(body_size, move(body));

        std::unique_lock<std::mutex> lk(_send_mtx);
        if (gain != nullptr && _send_gain_q.size() > 0) _send_gain_q.pop();
        if (mod != nullptr && mod->buffer.size() <= mod->_sent) {
          mod->_sent = 0;
          if (_send_mod_q.size() > 0) _send_mod_q.pop();
        }
      }
    } catch (const int errnum) {
      this->Close();
      std::cerr << errnum << "Link closed." << std::endl;
    }
  });
}

std::unique_ptr<uint8_t[]> AUTDController::MakeBody(GainPtr gain, ModulationPtr mod, size_t *size) {
  auto num_devices = (gain != nullptr) ? gain->geometry()->numDevices() : 0;

  *size = sizeof(RxGlobalHeader) + sizeof(uint16_t) * NUM_TRANS_IN_UNIT * num_devices;
  auto body = std::make_unique<uint8_t[]>(*size);

  auto *header = reinterpret_cast<RxGlobalHeader *>(&body[0]);
  header->msg_id = get_id();
  header->control_flags = 0;
  header->mod_size = 0;

  if (this->_silent_mode) header->control_flags |= SILENT;

  if (mod != nullptr) {
    const uint8_t mod_size = std::max(0, std::min(static_cast<int>(mod->buffer.size() - mod->_sent), MOD_FRAME_SIZE));
    header->mod_size = mod_size;
    if (mod->_sent == 0) header->control_flags |= LOOP_BEGIN;
    if (mod->_sent + mod_size >= mod->buffer.size()) header->control_flags |= LOOP_END;

    std::memcpy(header->mod, &mod->buffer[mod->_sent], mod_size);
    mod->_sent += mod_size;
  }

  auto *cursor = &body[0] + sizeof(RxGlobalHeader) / sizeof(body[0]);
  if (gain != nullptr) {
    for (int i = 0; i < gain->geometry()->numDevices(); i++) {
      auto deviceId = gain->geometry()->deviceIdForDeviceIdx(i);
      auto byteSize = NUM_TRANS_IN_UNIT * sizeof(uint16_t);
      std::memcpy(cursor, &gain->_data[deviceId].at(0), byteSize);
      cursor += byteSize / sizeof(body[0]);
    }
  }
  return body;
}

#pragma region deprecated
void AUTDController::AppendLateralGain(GainPtr gain) { this->AppendSTMGain(gain); }
void AUTDController::AppendLateralGain(const std::vector<GainPtr> &gain_list) { this->AppendSTMGain(gain_list); }
void AUTDController::StartLateralModulation(double freq) { this->StartSTModulation(freq); }
void AUTDController::FinishLateralModulation() { this->StopSTModulation(); }
void AUTDController::ResetLateralGain() { this->FinishSTModulation(); }
#pragma endregion

ControllerPtr Controller::Create() { return CreateHelper<AUTDController>(); }

}  // namespace autd
