/*
 * File: controller.cpp
 * Project: lib
 * Created Date: 13/05/2016
 * Author: Seki Inoue
 * -----
 * Last Modified: 18/02/2020
 * Modified By: Shun Suzuki (suzuki@hapis.k.u-tokyo.ac.jp)
 * -----
 * Copyright (c) 2016-2019 Hapis Lab. All rights reserved.
 *
 */

#include "controller.hpp"

#include <algorithm>
#include <chrono>
#include <condition_variable>
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
#pragma region Controller::impl
class Controller::impl {
 public:
  GeometryPtr _geometry;
  std::shared_ptr<internal::Link> _link;
  std::queue<GainPtr> _build_q;
  std::queue<GainPtr> _send_gain_q;
  std::queue<ModulationPtr> _send_mod_q;
  std::vector<GainPtr> _stmGains;
  std::vector<uint8_t *> _stmBodies;
  std::vector<size_t> _stmBodySizes;
  std::unique_ptr<Timer> _pStmTimer;

  std::thread _build_thr;
  std::thread _send_thr;
  std::condition_variable _build_cond;
  std::condition_variable _send_cond;
  std::mutex _build_mtx;
  std::mutex _send_mtx;

  bool silentMode = true;
  bool isOpen();

  impl();
  ~impl();
  void CalibrateModulation();
  void Close();

  void InitPipeline();
  void Stop();
  void AppendGain(const GainPtr gain);
  void AppendGainSync(const GainPtr gain);
  void AppendModulation(const ModulationPtr mod);
  void AppendModulationSync(const ModulationPtr mod);
  void AppendSTMGain(GainPtr gain);
  void AppendSTMGain(const std::vector<GainPtr> &gain_list);
  void StartSTModulation(float freq);
  void StopSTModulation();
  void FinishSTModulation();
  void FlushBuffer();

  std::unique_ptr<uint8_t[]> MakeBody(GainPtr gain, ModulationPtr mod, size_t *size);

  static uint8_t get_id() {
    static std::atomic<uint8_t> id{0};

    id.fetch_add(0x01);
    uint8_t expected = 0xf0;
    id.compare_exchange_weak(expected, 1);

    return id.load();
  }
};

Controller::impl::impl() {
  this->_geometry = Geometry::Create();
  this->silentMode = true;
  this->_pStmTimer = std::make_unique<Timer>();
}

Controller::impl::~impl() {
  if (std::this_thread::get_id() != this->_build_thr.get_id() && this->_build_thr.joinable()) this->_build_thr.join();
  if (std::this_thread::get_id() != this->_send_thr.get_id() && this->_send_thr.joinable()) this->_send_thr.join();
}

void Controller::impl::InitPipeline() {
  this->_build_thr = std::thread([&] {
    while (this->isOpen()) {
      GainPtr gain = nullptr;
      {
        std::unique_lock<std::mutex> lk(_build_mtx);

        _build_cond.wait(lk, [&] { return _build_q.size() || !this->isOpen(); });

        if (_build_q.size() > 0) {
          gain = _build_q.front();
          _build_q.pop();
        }
      }

      if (gain != nullptr) {
        if (!gain->built()) gain->build();
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
      while (this->isOpen()) {
        GainPtr gain = nullptr;
        ModulationPtr mod = nullptr;

        {
          std::unique_lock<std::mutex> lk(_send_mtx);
          _send_cond.wait(lk, [&] { return _send_gain_q.size() || _send_mod_q.size() || !this->isOpen(); });
          if (_send_gain_q.size() > 0) gain = _send_gain_q.front();
          if (_send_mod_q.size() > 0) mod = _send_mod_q.front();
        }
        size_t body_size = 0;
        auto body = MakeBody(gain, mod, &body_size);
        if (this->_link->isOpen()) this->_link->Send(body_size, move(body));

        std::unique_lock<std::mutex> lk(_send_mtx);
        if (gain != nullptr && _send_gain_q.size() > 0) _send_gain_q.pop();
        if (mod != nullptr && mod->buffer.size() <= mod->sent) {
          mod->sent = 0;
          if (_send_mod_q.size() > 0) _send_mod_q.pop();
        }
      }
    } catch (const int errnum) {
      this->Close();
      std::cerr << errnum << "Link closed." << std::endl;
    }
  });
}

void Controller::impl::Stop() {
  auto nullgain = NullGain::Create();
  this->AppendGainSync(nullgain);
#if DLL_FOR_CAPI
  delete nullgain;
#endif
}

void Controller::impl::AppendGain(GainPtr gain) {
  this->_pStmTimer->Stop();
  {
    gain->SetGeometry(this->_geometry);
    std::unique_lock<std::mutex> lk(_build_mtx);
    _build_q.push(gain);
  }
  _build_cond.notify_all();
}

void Controller::impl::AppendGainSync(GainPtr gain) {
  this->_pStmTimer->Stop();
  try {
    gain->SetGeometry(this->_geometry);
    if (!gain->built()) gain->build();

    size_t body_size = 0;
    auto body = this->MakeBody(gain, nullptr, &body_size);

    if (this->isOpen()) this->_link->Send(body_size, move(body));
  } catch (const int errnum) {
    this->_link->Close();
    std::cerr << errnum << "Link closed." << std::endl;
  }
}

void Controller::impl::AppendModulation(ModulationPtr mod) {
  std::unique_lock<std::mutex> lk(_send_mtx);
  _send_mod_q.push(mod);
  _send_cond.notify_all();
}

void Controller::impl::AppendModulationSync(ModulationPtr mod) {
  try {
    if (this->isOpen()) {
      while (mod->buffer.size() > mod->sent) {
        size_t body_size = 0;
        auto body = this->MakeBody(nullptr, mod, &body_size);
        this->_link->Send(body_size, move(body));
      }
      mod->sent = 0;
    }
  } catch (const int errnum) {
    this->Close();
    std::cerr << errnum << "Link closed." << std::endl;
  }
}

void Controller::impl::AppendSTMGain(GainPtr gain) { _stmGains.push_back(gain); }

void Controller::impl::AppendSTMGain(const std::vector<GainPtr> &gain_list) {
  for (auto g : gain_list) {
    this->AppendSTMGain(g);
  }
}

void Controller::impl::StartSTModulation(float freq) {
  auto len = this->_stmGains.size();
  auto itvl_us = static_cast<int>(1000000. / freq / len);
  this->_pStmTimer->SetInterval(itvl_us);

  auto current_size = this->_stmBodies.size();
  this->_stmBodies.resize(len);
  this->_stmBodySizes.resize(len);

  for (size_t i = current_size; i < len; i++) {
    auto g = this->_stmGains[i];
    g->SetGeometry(this->_geometry);
    if (!g->built()) g->build();

    size_t body_size = 0;
    auto body = this->MakeBody(g, nullptr, &body_size);
    uint8_t *b = new uint8_t[body_size];
    std::memcpy(b, body.get(), body_size);
    this->_stmBodies[i] = b;
    this->_stmBodySizes[i] = body_size;
  }

  size_t idx = 0;
  this->_pStmTimer->Start([this, idx, len]() mutable {
    auto body_size = this->_stmBodySizes[idx];
    auto body_copy = std::make_unique<uint8_t[]>(body_size);
    uint8_t *p = this->_stmBodies[idx];
    std::memcpy(body_copy.get(), p, body_size);
    if (this->isOpen()) this->_link->Send(body_size, std::move(body_copy));
    idx = (idx + 1) % len;
  });
}

void Controller::impl::StopSTModulation() {
  this->_pStmTimer->Stop();
  this->Stop();
}

void Controller::impl::FinishSTModulation() {
  this->StopSTModulation();
  std::vector<GainPtr>().swap(this->_stmGains);
  for (uint8_t *p : this->_stmBodies) {
    delete[] p;
  }
  std::vector<uint8_t *>().swap(this->_stmBodies);
  std::vector<size_t>().swap(this->_stmBodySizes);
}

void Controller::impl::CalibrateModulation() { this->_link->CalibrateModulation(); }

void Controller::impl::FlushBuffer() {
  std::unique_lock<std::mutex> lk0(_send_mtx);
  std::unique_lock<std::mutex> lk1(_build_mtx);
  std::queue<GainPtr>().swap(_build_q);
  std::queue<GainPtr>().swap(_send_gain_q);
  std::queue<ModulationPtr>().swap(_send_mod_q);
}

std::unique_ptr<uint8_t[]> Controller::impl::MakeBody(GainPtr gain, ModulationPtr mod, size_t *size) {
  auto num_devices = (gain != nullptr) ? gain->geometry()->numDevices() : 0;

  *size = sizeof(RxGlobalHeader) + sizeof(uint16_t) * NUM_TRANS_IN_UNIT * num_devices;
  auto body = std::make_unique<uint8_t[]>(*size);

  auto *header = reinterpret_cast<RxGlobalHeader *>(&body[0]);
  header->msg_id = get_id();
  header->control_flags = 0;
  header->mod_size = 0;

  if (this->silentMode) header->control_flags |= SILENT;

  if (mod != nullptr) {
    const uint8_t mod_size = std::max(0, std::min(static_cast<int>(mod->buffer.size() - mod->sent), MOD_FRAME_SIZE));
    header->mod_size = mod_size;
    if (mod->sent == 0) header->control_flags |= LOOP_BEGIN;
    if (mod->sent + mod_size >= mod->buffer.size()) header->control_flags |= LOOP_END;
    header->frequency_shift = this->_geometry->_freq_shift;

    std::memcpy(header->mod, &mod->buffer[mod->sent], mod_size);
    mod->sent += mod_size;
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

bool Controller::impl::isOpen() { return this->_link.get() && this->_link->isOpen(); }

void Controller::impl::Close() {
  if (this->isOpen()) {
    this->FinishSTModulation();
    this->Stop();
    this->_link->Close();
    this->FlushBuffer();
    this->_build_cond.notify_all();
    if (std::this_thread::get_id() != this->_build_thr.get_id() && this->_build_thr.joinable()) this->_build_thr.join();
    this->_send_cond.notify_all();
    if (std::this_thread::get_id() != this->_send_thr.get_id() && this->_send_thr.joinable()) this->_send_thr.join();
    this->_link = std::shared_ptr<internal::Link>(nullptr);
  }
}
#pragma endregion

#pragma region pimpl
Controller::Controller() { this->_pimpl = std::make_unique<impl>(); }

Controller::~Controller() noexcept(false) { this->Close(); }

bool Controller::isOpen() { return this->_pimpl->isOpen(); }

GeometryPtr Controller::geometry() noexcept { return this->_pimpl->_geometry; }

bool Controller::silentMode() noexcept { return this->_pimpl->silentMode; }

size_t Controller::remainingInBuffer() {
  return this->_pimpl->_send_gain_q.size() + this->_pimpl->_send_mod_q.size() + this->_pimpl->_build_q.size();
}

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
#if DLL_FOR_CAPI
    p.first = *adapter.desc.get();
    p.second = *adapter.name.get();
    res[i++] = p;
#else
    p.first = adapter.desc;
    p.second = adapter.name;
    res.push_back(p);
#endif
  }
  return res;
}

void Controller::Open(LinkType type, std::string location) {
  this->Close();

  switch (type) {
#if WIN32
    case LinkType::ETHERCAT:
    case LinkType::TwinCAT: {
      // TODO(volunteer): a smarter localhost detection
      if (location == "" || location.find("localhost") == 0 || location.find("0.0.0.0") == 0 || location.find("127.0.0.1") == 0) {
        this->_pimpl->_link = std::make_shared<internal::LocalEthercatLink>();
      } else {
        this->_pimpl->_link = std::make_shared<internal::EthercatLink>();
      }
      this->_pimpl->_link->Open(location);
      break;
    }
#endif
    case LinkType::SOEM: {
      this->_pimpl->_link = std::make_shared<internal::SOEMLink>();
      auto devnum = this->_pimpl->_geometry->numDevices();
      this->_pimpl->_link->Open(location + ":" + std::to_string(devnum));
      break;
    }
    default:
      throw std::runtime_error("This link type is not implemented yet.");
      break;
  }

  if (this->_pimpl->_link->isOpen())
    this->_pimpl->InitPipeline();
  else
    this->Close();
}

void Controller::SetSilentMode(bool silent) noexcept { this->_pimpl->silentMode = silent; }

void Controller::CalibrateModulation() { this->_pimpl->CalibrateModulation(); }

void Controller::Close() { this->_pimpl->Close(); }

void Controller::Stop() { this->_pimpl->Stop(); }

void Controller::AppendGain(GainPtr gain) { this->_pimpl->AppendGain(gain); }

void Controller::AppendGainSync(GainPtr gain) { this->_pimpl->AppendGainSync(gain); }

void Controller::AppendModulation(ModulationPtr modulation) { this->_pimpl->AppendModulation(modulation); }

void Controller::AppendModulationSync(ModulationPtr modulation) { this->_pimpl->AppendModulationSync(modulation); }

void Controller::AppendSTMGain(GainPtr gain) { this->_pimpl->AppendSTMGain(gain); }

void Controller::AppendSTMGain(const std::vector<GainPtr> &gain_list) { this->_pimpl->AppendSTMGain(gain_list); }

void Controller::StartSTModulation(float freq) { this->_pimpl->StartSTModulation(freq); }

void Controller::StopSTModulation() { this->_pimpl->StopSTModulation(); }

void Controller::FinishSTModulation() { this->_pimpl->FinishSTModulation(); }

void Controller::Flush() { this->_pimpl->FlushBuffer(); }

void Controller::LateralModulationAT(Eigen::Vector3f point, Eigen::Vector3f dir, float lm_amp, float lm_freq) {
  auto p1 = point + lm_amp * dir;
  auto p2 = point - lm_amp * dir;
  this->FinishSTModulation();
  this->AppendSTMGain(autd::FocalPointGain::Create(p1));
  this->AppendSTMGain(autd::FocalPointGain::Create(p2));
  this->StartSTModulation(lm_freq);
}

#pragma region deprecated
void Controller::AppendLateralGain(GainPtr gain) { this->AppendSTMGain(gain); }
void Controller::AppendLateralGain(const std::vector<GainPtr> &gain_list) { this->AppendSTMGain(gain_list); }
void Controller::StartLateralModulation(float freq) { this->StartSTModulation(freq); }
void Controller::FinishLateralModulation() { this->StopSTModulation(); }
void Controller::ResetLateralGain() { this->FinishSTModulation(); }
#pragma endregion

#pragma endregion
}  // namespace autd
