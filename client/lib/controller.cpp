// File: controller.cpp
// Project: lib
// Created Date: 13/05/2016
// Author: Seki Inoue
// -----
// Last Modified: 22/12/2020
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
ControllerPtr Controller::Create() { return std::make_shared<_internal::AUTDController>(); }
}  // namespace autd

namespace autd::_internal {

using std::move;
using std::thread, std::queue;
using std::unique_ptr, std::shared_ptr;
using std::vector, std::condition_variable, std::unique_lock, std::mutex;

class AUTDControllerSync {
 public:
  explicit AUTDControllerSync(shared_ptr<AUTDLogic> logic) : _autd_logic(logic) {}

  void AppendGain(GainPtr gain, bool wait_for_send) {
    this->_autd_logic->_seq_mode = false;
    gain->SetGeometry(this->_autd_logic->_geometry);
    gain->Build();
    this->_autd_logic->SendBlocking(gain, nullptr);
  }
  void AppendModulation(ModulationPtr mod) {
    mod->Build(this->_autd_logic->_config);
    while (mod->buffer.size() > mod->sent()) {
      _autd_logic->SendBlocking(nullptr, mod);
    }
    mod->reset();
  }
  void AppendSeq(SequencePtr seq) {
    this->_autd_logic->_seq_mode = true;
    while (seq->sent() < seq->control_points().size()) {
      this->_autd_logic->SendBlocking(seq);
    }
    this->_autd_logic->CalibrateSeq();
  }

 private:
  shared_ptr<AUTDLogic> _autd_logic;
};

class AUTDControllerAsync {
 public:
  explicit AUTDControllerAsync(shared_ptr<AUTDLogic> logic) : _autd_logic(logic) { this->_is_running = false; }

  ~AUTDControllerAsync() {
    if (std::this_thread::get_id() != this->_build_gain_thr.get_id() && this->_build_gain_thr.joinable()) this->_build_gain_thr.join();
    if (std::this_thread::get_id() != this->_build_mod_thr.get_id() && this->_build_mod_thr.joinable()) this->_build_mod_thr.join();
    if (std::this_thread::get_id() != this->_send_thr.get_id() && this->_send_thr.joinable()) this->_send_thr.join();
  }

  size_t remaining_in_buffer() {
    return this->_send_gain_q.size() + this->_send_mod_q.size() + this->_build_gain_q.size() + this->_build_mod_q.size();
  }

  void Flush() {
    unique_lock<mutex> lk0(_send_mtx);
    unique_lock<mutex> lk1(_build_gain_mtx);
    unique_lock<mutex> lk2(_build_mod_mtx);
    queue<GainPtr>().swap(_build_gain_q);
    queue<ModulationPtr>().swap(_build_mod_q);
    queue<GainPtr>().swap(_send_gain_q);
    queue<ModulationPtr>().swap(_send_mod_q);
  }

  void Close() {
    this->Flush();
    this->_is_running = false;
    this->_build_gain_cond.notify_all();
    if (std::this_thread::get_id() != this->_build_gain_thr.get_id() && this->_build_gain_thr.joinable()) this->_build_gain_thr.join();
    this->_build_mod_cond.notify_all();
    if (std::this_thread::get_id() != this->_build_mod_thr.get_id() && this->_build_mod_thr.joinable()) this->_build_mod_thr.join();
    this->_send_cond.notify_all();
    if (std::this_thread::get_id() != this->_send_thr.get_id() && this->_send_thr.joinable()) this->_send_thr.join();
  }

  void AppendGain(GainPtr gain) {
    gain->SetGeometry(this->_autd_logic->_geometry);
    {
      std::unique_lock<std::mutex> lk(_build_gain_mtx);
      _build_gain_q.push(gain);
    }
    _build_gain_cond.notify_all();
  }
  void AppendModulation(ModulationPtr mod) {
    {
      std::unique_lock<std::mutex> lk(_build_mod_mtx);
      _build_mod_q.push(mod);
    }
    _build_mod_cond.notify_all();
  }

  void AUTDControllerAsync::InitPipeline() {
    this->_is_running = true;
    this->_build_gain_thr = std::thread([&] {
      while (this->_is_running) {
        GainPtr gain = nullptr;
        {
          std::unique_lock<std::mutex> lk(_build_gain_mtx);

          _build_gain_cond.wait(lk, [&] { return _build_gain_q.size() || !this->_is_running; });

          if (_build_gain_q.size() > 0) {
            gain = _build_gain_q.front();
            _build_gain_q.pop();
          }
        }

        if (gain != nullptr) {
          gain->SetGeometry(this->_autd_logic->_geometry);
          gain->Build();
          {
            std::unique_lock<std::mutex> lk(_send_mtx);
            _send_gain_q.push(gain);
            _send_cond.notify_all();
          }
        }
      }
    });

    this->_build_mod_thr = std::thread([&] {
      while (this->_is_running) {
        ModulationPtr mod = nullptr;
        {
          std::unique_lock<std::mutex> lk(_build_mod_mtx);

          _build_mod_cond.wait(lk, [&] { return _build_mod_q.size() || !_is_running; });

          if (_build_mod_q.size() > 0) {
            mod = _build_mod_q.front();
            _build_mod_q.pop();
          }
        }

        if (mod != nullptr) {
          mod->Build(this->_autd_logic->_config);
          {
            std::unique_lock<std::mutex> lk(_send_mtx);
            _send_mod_q.push(mod);
            _send_cond.notify_all();
          }
        }
      }
    });

    this->_send_thr = std::thread([&] {
      while (this->_is_running) {
        GainPtr gain = nullptr;
        ModulationPtr mod = nullptr;

        {
          std::unique_lock<std::mutex> lk(_send_mtx);
          _send_cond.wait(lk, [&] { return _send_gain_q.size() || _send_mod_q.size() || !this->_is_running; });
          if (_send_gain_q.size() > 0) gain = _send_gain_q.front();
          if (_send_mod_q.size() > 0) mod = _send_mod_q.front();
        }
        this->_autd_logic->SendBlocking(gain, mod);

        std::unique_lock<std::mutex> lk(_send_mtx);
        if (gain != nullptr && _send_gain_q.size() > 0) _send_gain_q.pop();
        if (mod != nullptr && mod->buffer.size() <= mod->sent()) {
          mod->reset();
          _send_mod_q.pop();
        }
      }
    });
  }

 private:
  bool _is_running;
  shared_ptr<AUTDLogic> _autd_logic;

  queue<GainPtr> _build_gain_q;
  queue<ModulationPtr> _build_mod_q;
  queue<GainPtr> _send_gain_q;
  queue<ModulationPtr> _send_mod_q;

  thread _build_gain_thr;
  thread _build_mod_thr;
  thread _send_thr;
  condition_variable _build_gain_cond;
  condition_variable _build_mod_cond;
  condition_variable _send_cond;
  mutex _build_gain_mtx;
  mutex _build_mod_mtx;
  mutex _send_mtx;
};

class AUTDControllerSTM {
 public:
  explicit AUTDControllerSTM(shared_ptr<AUTDLogic> logic) : _autd_logic(logic) { this->_p_stm_timer = std::make_unique<Timer>(); }

  ~AUTDControllerSTM() {}

  void AppendGain(GainPtr gain) { _stm_gains.push_back(gain); }
  void AppendGain(const std::vector<GainPtr>& gain_list) {
    for (GainPtr g : gain_list) {
      this->AppendGain(g);
    }
  }

  void Start(double freq) {
    auto len = this->_stm_gains.size();
    auto itvl_us = static_cast<int>(1000000. / freq / len);
    this->_p_stm_timer->SetInterval(itvl_us);

    auto current_size = this->_stm_bodies.size();
    this->_stm_bodies.resize(len);
    this->_stm_body_sizes.resize(len);

    for (size_t i = current_size; i < len; i++) {
      GainPtr g = this->_stm_gains[i];
      g->SetGeometry(this->_autd_logic->_geometry);
      g->Build();

      size_t body_size = 0;
      uint8_t msg_id = 0;
      auto body = this->_autd_logic->MakeBody(g, nullptr, &body_size, &msg_id);
      uint8_t* b = new uint8_t[body_size];
      std::memcpy(b, body.get(), body_size);
      this->_stm_bodies[i] = b;
      this->_stm_body_sizes[i] = body_size;
    }

    size_t idx = 0;
    this->_p_stm_timer->Start([this, idx, len]() mutable {
      auto body_size = this->_stm_body_sizes[idx];
      auto body_copy = std::make_unique<uint8_t[]>(body_size);
      uint8_t* p = this->_stm_bodies[idx];
      std::memcpy(body_copy.get(), p, body_size);
      this->_autd_logic->SendData(body_size, move(body_copy));
      idx = (idx + 1) % len;
    });
  }

  void Stop() { this->_p_stm_timer->Stop(); }

  void Finish() {
    this->Stop();
    vector<GainPtr>().swap(this->_stm_gains);
    for (uint8_t* p : this->_stm_bodies) {
      delete[] p;
    }
    vector<uint8_t*>().swap(this->_stm_bodies);
    vector<size_t>().swap(this->_stm_body_sizes);
  }

  void Close() { this->Finish(); }

  shared_ptr<AUTDLogic> _autd_logic;

  vector<GainPtr> _stm_gains;
  vector<uint8_t*> _stm_bodies;
  vector<size_t> _stm_body_sizes;
  unique_ptr<Timer> _p_stm_timer;
};

AUTDController::AUTDController() {
  this->_autd_logic = std::make_shared<AUTDLogic>();
  this->_sync_cnt = std::make_unique<AUTDControllerSync>(this->_autd_logic);
  this->_async_cnt = std::make_unique<AUTDControllerAsync>(this->_autd_logic);
  this->_stm_cnt = std::make_unique<AUTDControllerSTM>(this->_autd_logic);
}

AUTDController::~AUTDController() {}

bool AUTDController::is_open() { return this->_autd_logic->is_open(); }

GeometryPtr AUTDController::geometry() noexcept { return this->_autd_logic->_geometry; }

bool AUTDController::silent_mode() noexcept { return this->_autd_logic->_silent_mode; }

size_t AUTDController::remaining_in_buffer() { return this->_async_cnt->remaining_in_buffer(); }

void AUTDController::SetSilentMode(bool silent) noexcept { this->_autd_logic->_silent_mode = silent; }

void AUTDController::OpenWith(LinkPtr link) {
  this->Close();
  this->_autd_logic->OpenWith(link);
  if (this->_autd_logic->is_open()) this->_async_cnt->InitPipeline();
}

bool AUTDController::Calibrate(Configuration config) { return this->_autd_logic->Calibrate(config); }

bool AUTDController::Clear() { return this->_autd_logic->Clear(); }

void AUTDController::Close() {
  this->_stm_cnt->Close();
  this->_async_cnt->Close();
  this->Stop();
  this->Clear();
  this->_autd_logic->Close();
}

void AUTDController::Flush() { this->_async_cnt->Flush(); }

void AUTDController::Stop() {
  auto nullgain = autd::gain::NullGain::Create();
  this->AppendGainSync(nullgain, true);
}

void AUTDController::AppendGain(GainPtr gain) {
  this->_stm_cnt->Stop();
  this->_async_cnt->AppendGain(gain);
}
void AUTDController::AppendGainSync(GainPtr gain, bool wait_for_send) {
  this->_stm_cnt->Stop();
  this->_sync_cnt->AppendGain(gain, wait_for_send);
}

void AUTDController::AppendModulation(ModulationPtr mod) { this->_sync_cnt->AppendModulation(mod); }
void AUTDController::AppendModulationSync(ModulationPtr mod) { this->_sync_cnt->AppendModulation(mod); }

void AUTDController::AppendSTMGain(GainPtr gain) { this->_stm_cnt->AppendGain(gain); }
void AUTDController::AppendSTMGain(const std::vector<GainPtr>& gain_list) { this->_stm_cnt->AppendGain(gain_list); }
void AUTDController::StartSTModulation(double freq) { this->_stm_cnt->Start(freq); }
void AUTDController::StopSTModulation() {
  this->_stm_cnt->Stop();
  this->Stop();
}
void AUTDController::FinishSTModulation() { this->_stm_cnt->Finish(); }

void AUTDController::LateralModulationAT(Vector3 point, Vector3 dir, double lm_amp, double lm_freq) {
  auto p1 = point + lm_amp * dir;
  auto p2 = point - lm_amp * dir;
  this->FinishSTModulation();
  this->AppendSTMGain(autd::gain::FocalPointGain::Create(p1));
  this->AppendSTMGain(autd::gain::FocalPointGain::Create(p2));
  this->StartSTModulation(lm_freq);
}

void AUTDController::AppendSequence(SequencePtr seq) { this->_sync_cnt->AppendSeq(seq); }

FirmwareInfoList AUTDController::firmware_info_list() { return this->_autd_logic->firmware_info_list(); }

}  // namespace autd::_internal
