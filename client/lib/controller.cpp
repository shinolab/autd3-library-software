// File: controller.cpp
// Project: lib
// Created Date: 13/05/2016
// Author: Seki Inoue
// -----
// Last Modified: 03/04/2021
// Modified By: Shun Suzuki (suzuki@hapis.k.u-tokyo.ac.jp)
// -----
// Copyright (c) 2016-2020 Hapis Lab. All rights reserved.
//

#include "controller.hpp"

#include <algorithm>
#include <condition_variable>
#include <cstring>
#include <mutex>
#include <queue>
#include <thread>
#include <utility>
#include <vector>

#include "controller_impl.hpp"
#include "firmware_version.hpp"
#include "link.hpp"
#include "sequence.hpp"
#include "timer.hpp"

namespace autd {
namespace internal {

using std::move;
using std::thread, std::queue;
using std::unique_ptr, std::shared_ptr;
using std::vector, std::condition_variable, std::unique_lock, std::mutex;

class AUTDControllerSync {
 public:
  explicit AUTDControllerSync(const shared_ptr<AUTDLogic>& logic) : _autd_logic(logic) {}

  [[nodiscard]] Result<bool, std::string> AppendGain(const GainPtr& gain, const bool wait_for_send) const {
    this->_autd_logic->BuildGain(gain);
    if (wait_for_send) return this->_autd_logic->SendBlocking(gain, nullptr);

    return this->_autd_logic->Send(gain, nullptr);
  }

  [[nodiscard]] Result<bool, std::string> AppendModulation(const ModulationPtr& mod) const {
    auto success = true;
    this->_autd_logic->BuildModulation(mod);
    while (mod->buffer.size() > mod->sent()) {
      auto res = _autd_logic->SendBlocking(nullptr, mod);
      if (res.is_err()) return res;
      success &= res.unwrap();
    }
    mod->sent() = 0;
    return Ok(success);
  }

  [[nodiscard]] Result<bool, std::string> AppendSeq(const SequencePtr& seq) const {
    auto success = true;
    while (seq->sent() < seq->control_points().size()) {
      auto res = this->_autd_logic->SendBlocking(seq);
      if (res.is_err()) return res;
      success &= res.unwrap();
    }
    auto sync_res = this->_autd_logic->SynchronizeSeq();
    if (sync_res.is_err()) return sync_res;

    return Ok(success && sync_res.unwrap());
  }

 private:
  shared_ptr<AUTDLogic> _autd_logic;
};

class AUTDControllerAsync {
 public:
  explicit AUTDControllerAsync(const shared_ptr<AUTDLogic>& logic) : _autd_logic(logic) { this->_is_running = false; }
  AUTDControllerAsync(const AUTDControllerAsync& obj) = delete;
  AUTDControllerAsync(AUTDControllerAsync&& obj) = delete;
  AUTDControllerAsync& operator=(const AUTDControllerAsync& obj) = delete;
  AUTDControllerAsync& operator=(AUTDControllerAsync&& obj) = delete;

  ~AUTDControllerAsync() {
    if (std::this_thread::get_id() != this->_build_gain_thr.get_id() && this->_build_gain_thr.joinable()) this->_build_gain_thr.join();
    if (std::this_thread::get_id() != this->_build_mod_thr.get_id() && this->_build_mod_thr.joinable()) this->_build_mod_thr.join();
    if (std::this_thread::get_id() != this->_send_thr.get_id() && this->_send_thr.joinable()) this->_send_thr.join();
  }

  [[nodiscard]] size_t remaining_in_buffer() const {
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

  void AppendGain(const GainPtr& gain) {
    {
      unique_lock<mutex> lk(_build_gain_mtx);
      _build_gain_q.push(gain);
    }
    _build_gain_cond.notify_all();
  }
  void AppendModulation(const ModulationPtr& mod) {
    {
      unique_lock<mutex> lk(_build_mod_mtx);
      _build_mod_q.push(mod);
    }
    _build_mod_cond.notify_all();
  }

  void InitPipeline() {
    this->_is_running = true;
    this->_build_gain_thr = std::thread([&] {
      while (this->_is_running) {
        GainPtr gain = nullptr;
        {
          unique_lock<mutex> lk(_build_gain_mtx);

          _build_gain_cond.wait(lk, [&] { return !_build_gain_q.empty() || !this->_is_running; });

          if (!_build_gain_q.empty()) {
            gain = _build_gain_q.front();
            _build_gain_q.pop();
          }
        }

        if (gain != nullptr) {
          this->_autd_logic->BuildGain(gain);
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
          unique_lock<mutex> lk(_build_mod_mtx);

          _build_mod_cond.wait(lk, [&] { return !_build_mod_q.empty() || !_is_running; });

          if (!_build_mod_q.empty()) {
            mod = _build_mod_q.front();
            _build_mod_q.pop();
          }
        }

        if (mod != nullptr) {
          this->_autd_logic->BuildModulation(mod);
          {
            unique_lock<mutex> lk(_send_mtx);
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
          unique_lock<mutex> lk(_send_mtx);
          _send_cond.wait(lk, [&] { return !_send_gain_q.empty() || !_send_mod_q.empty() || !this->_is_running; });
          if (!_send_gain_q.empty()) gain = _send_gain_q.front();
          if (!_send_mod_q.empty()) mod = _send_mod_q.front();
        }
        this->_autd_logic->Send(gain, mod);

        unique_lock<mutex> lk(_send_mtx);
        if (gain != nullptr) _send_gain_q.pop();
        if (mod != nullptr && mod->buffer.size() <= mod->sent()) {
          mod->sent() = 0;
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

class AUTDControllerStm {
 public:
  explicit AUTDControllerStm(const shared_ptr<AUTDLogic>& logic) : _autd_logic(logic), _stm_timer() {}
  AUTDControllerStm(const AUTDControllerStm& obj) = delete;
  AUTDControllerStm(AUTDControllerStm&& obj) = delete;
  AUTDControllerStm& operator=(const AUTDControllerStm& obj) = delete;
  AUTDControllerStm& operator=(AUTDControllerStm&& obj) = delete;

  ~AUTDControllerStm() = default;

  void AppendGain(const GainPtr& gain) { _stm_gains.emplace_back(gain); }
  void AppendGain(const vector<GainPtr>& gain_list) {
    for (const auto& g : gain_list) {
      this->AppendGain(g);
    }
  }

  [[nodiscard]] Result<bool, std::string> Start(const Float freq) {
    auto len = this->_stm_gains.size();
    auto interval_us = static_cast<uint32_t>(1000000. / static_cast<double>(freq) / static_cast<double>(len));
    this->_stm_timer.SetInterval(interval_us);

    const auto current_size = this->_stm_bodies.size();
    this->_stm_bodies.resize(len);
    this->_stm_body_sizes.resize(len);

    for (auto i = current_size; i < len; i++) {
      auto& g = this->_stm_gains[i];
      this->_autd_logic->BuildGain(g);

      size_t body_size = 0;
      uint8_t msg_id = 0;
      auto body = this->_autd_logic->MakeBody(g, nullptr, &body_size, &msg_id);
      auto* const b = new uint8_t[body_size];
      std::memcpy(b, body.get(), body_size);
      this->_stm_bodies[i] = b;
      this->_stm_body_sizes[i] = body_size;
    }

    size_t idx = 0;
    return this->_stm_timer.Start([this, idx, len]() mutable {
      const auto body_size = this->_stm_body_sizes[idx];
      auto body_copy = std::make_unique<uint8_t[]>(body_size);
      auto* const p = this->_stm_bodies[idx];
      std::memcpy(body_copy.get(), p, body_size);
      this->_autd_logic->SendData(body_size, move(body_copy));
      idx = (idx + 1) % len;
    });
  }

  [[nodiscard]] Result<bool, std::string> Stop() { return this->_stm_timer.Stop(); }

  [[nodiscard]] Result<bool, std::string> Finish() {
    this->Stop();
    vector<GainPtr>().swap(this->_stm_gains);
    for (auto* p : this->_stm_bodies) {
      delete[] p;
    }
    vector<uint8_t*>().swap(this->_stm_bodies);
    vector<size_t>().swap(this->_stm_body_sizes);

    return Ok(true);
  }

  [[nodiscard]] Result<bool, std::string> Close() { return this->Finish(); }

 private:
  shared_ptr<AUTDLogic> _autd_logic;

  vector<GainPtr> _stm_gains;
  vector<uint8_t*> _stm_bodies;
  vector<size_t> _stm_body_sizes;
  Timer _stm_timer;
};

AUTDController::AUTDController() {
  this->_autd_logic = std::make_shared<AUTDLogic>();
  this->_sync_cnt = std::make_unique<AUTDControllerSync>(this->_autd_logic);
  this->_async_cnt = std::make_unique<AUTDControllerAsync>(this->_autd_logic);
  this->_stm_cnt = std::make_unique<AUTDControllerStm>(this->_autd_logic);
}

AUTDController::~AUTDController() = default;

bool AUTDController::is_open() { return this->_autd_logic->is_open(); }

GeometryPtr AUTDController::geometry() noexcept { return this->_autd_logic->geometry(); }

bool AUTDController::silent_mode() noexcept { return this->_autd_logic->silent_mode(); }

size_t AUTDController::remaining_in_buffer() { return this->_async_cnt->remaining_in_buffer(); }

void AUTDController::SetSilentMode(const bool silent) noexcept { this->_autd_logic->silent_mode() = silent; }

Result<bool, std::string> AUTDController::OpenWith(LinkPtr link) {
  this->Close();
  const auto res = this->_autd_logic->OpenWith(move(link));
  if (res.is_err()) return res;

  this->_async_cnt->InitPipeline();
  return Ok(true);
}
Result<bool, std::string> AUTDController::Calibrate(const Configuration config) { return Synchronize(config); }
Result<bool, std::string> AUTDController::Synchronize(const Configuration config) { return this->_autd_logic->Synchronize(config); }

Result<bool, std::string> AUTDController::Clear() { return this->_autd_logic->Clear(); }

Result<bool, std::string> AUTDController::Close() {
  this->_stm_cnt->Close();
  this->_async_cnt->Close();

  auto stop_res = this->Stop();
  if (stop_res.is_err()) return stop_res;

  auto clear_res = this->Clear();
  if (clear_res.is_err()) return clear_res;

  auto close_res = this->_autd_logic->Close();
  if (close_res.is_err()) return close_res;

  return Ok(stop_res.unwrap_or(false) && clear_res.unwrap_or(false) && close_res.unwrap());
}

void AUTDController::Flush() { this->_async_cnt->Flush(); }

Result<bool, std::string> AUTDController::Stop() {
  const auto null_gain = gain::NullGain::Create();
  return this->AppendGainSync(null_gain, true);
}

void AUTDController::AppendGain(const GainPtr gain) {
  this->_stm_cnt->Stop();
  this->_async_cnt->AppendGain(gain);
}
Result<bool, std::string> AUTDController::AppendGainSync(const GainPtr gain, const bool wait_for_send) {
  const auto stm_stop = this->_stm_cnt->Stop();
  if (stm_stop.is_err()) return stm_stop;

  return this->_sync_cnt->AppendGain(gain, wait_for_send);
}

void AUTDController::AppendModulation(const ModulationPtr mod) { this->_sync_cnt->AppendModulation(mod); }
Result<bool, std::string> AUTDController::AppendModulationSync(const ModulationPtr mod) { return this->_sync_cnt->AppendModulation(mod); }

void AUTDController::AppendSTMGain(const GainPtr gain) { this->_stm_cnt->AppendGain(gain); }
void AUTDController::AppendSTMGain(const std::vector<GainPtr>& gain_list) { this->_stm_cnt->AppendGain(gain_list); }
Result<bool, std::string> AUTDController::StartSTModulation(const Float freq) { return this->_stm_cnt->Start(freq); }
void AUTDController::StopSTModulation() {
  this->_stm_cnt->Stop();
  this->Stop();
}
void AUTDController::FinishSTModulation() { this->_stm_cnt->Finish(); }

Result<bool, std::string> AUTDController::AppendSequence(const SequencePtr seq) { return this->_sync_cnt->AppendSeq(seq); }

Result<std::vector<FirmwareInfo>, std::string> AUTDController::firmware_info_list() { return this->_autd_logic->firmware_info_list(); }
}  // namespace internal

ControllerPtr Controller::Create() { return std::make_unique<internal::AUTDController>(); }

}  // namespace autd
