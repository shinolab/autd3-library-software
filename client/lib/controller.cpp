// File: controller.cpp
// Project: lib
// Created Date: 13/05/2016
// Author: Seki Inoue
// -----
// Last Modified: 18/08/2020
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

#include "ec_config.hpp"
#include "emulator_link.hpp"
#include "firmware_version.hpp"
#include "geometry.hpp"
#include "link.hpp"
#include "privdef.hpp"
#include "sequence.hpp"
#include "timer.hpp"

namespace autd {
class AUTDController : public Controller {
 public:
  AUTDController();
  ~AUTDController() override;

  bool is_open() final;
  GeometryPtr geometry() noexcept final;
  bool silent_mode() noexcept final;
  size_t remaining_in_buffer() final;

  void OpenWith(LinkPtr link) final;
  void SetSilentMode(bool silent) noexcept final;
  bool Calibrate() final;
  bool Clear() final;
  void Close() final;

  void Stop() final;
  void AppendGain(const GainPtr gain) final;
  void AppendGainSync(const GainPtr gain, bool wait_for_send = false) final;
  void AppendModulation(const ModulationPtr mod) final;
  void AppendModulationSync(const ModulationPtr mod) final;
  void AppendSTMGain(GainPtr gain) final;
  void AppendSTMGain(const std::vector<GainPtr> &gain_list) final;
  void StartSTModulation(double freq) final;
  void StopSTModulation() final;
  void FinishSTModulation() final;
  void AppendSequence(SequencePtr seq) final;
  void Flush() final;
  FirmwareInfoList firmware_info_list() final;

  void LateralModulationAT(Vector3 point, Vector3 dir, double lm_amp = 2.5, double lm_freq = 100) final;

 private:
  GeometryPtr _geometry;
  LinkPtr _link;
  std::queue<GainPtr> _build_q;
  std::queue<GainPtr> _send_gain_q;
  std::queue<ModulationPtr> _send_mod_q;

  std::vector<GainPtr> _stm_gains;
  std::vector<uint8_t *> _stm_bodies;
  std::vector<size_t> _stm_body_sizes;
  std::unique_ptr<Timer> _p_stm_timer;

  std::thread _build_thr;
  std::thread _send_thr;
  std::condition_variable _build_cond;
  std::condition_variable _send_cond;
  std::mutex _build_mtx;
  std::mutex _send_mtx;

  std::vector<uint8_t> _rx_data;

  bool _silent_mode = true;

  void InitPipeline();
  std::unique_ptr<uint8_t[]> MakeBody(GainPtr gain, ModulationPtr mod, size_t *const size, uint8_t *const send_msg_id);
  bool WaitMsgProcessed(uint8_t msg_id, size_t max_trial = 200, uint8_t mask = 0xFF);

  std::unique_ptr<uint8_t[]> MakeSeqBody(SequencePtr seq, size_t *const size, uint8_t *const send_msg_id);
  void CalibrateSeq();
  std::unique_ptr<uint8_t[]> MakeCalibBody(std::vector<uint16_t> diffs, size_t *const size);

  static uint8_t get_id() {
    static std::atomic<uint8_t> id{OP_MODE_MSG_ID_MIN - 1};

    id.fetch_add(0x01);
    uint8_t expected = OP_MODE_MSG_ID_MAX + 1;
    id.compare_exchange_weak(expected, OP_MODE_MSG_ID_MIN);

    return id.load();
  }
};

AUTDController::AUTDController() {
  this->_link = nullptr;
  this->_geometry = Geometry::Create();
  this->_silent_mode = true;
  this->_p_stm_timer = std::make_unique<Timer>();
}

AUTDController::~AUTDController() {
  if (std::this_thread::get_id() != this->_build_thr.get_id() && this->_build_thr.joinable()) this->_build_thr.join();
  if (std::this_thread::get_id() != this->_send_thr.get_id() && this->_send_thr.joinable()) this->_send_thr.join();
}

bool AUTDController::is_open() { return this->_link != nullptr && this->_link->is_open(); }

GeometryPtr AUTDController::geometry() noexcept { return this->_geometry; }

bool AUTDController::silent_mode() noexcept { return this->_silent_mode; }

size_t AUTDController::remaining_in_buffer() { return this->_send_gain_q.size() + this->_send_mod_q.size() + this->_build_q.size(); }

void AUTDController::OpenWith(LinkPtr link) {
  this->Close();

  this->_link = link;
  this->_link->Open();
  if (this->_link->is_open())
    this->InitPipeline();
  else
    this->Close();
}

void AUTDController::SetSilentMode(bool silent) noexcept { this->_silent_mode = silent; }

bool AUTDController::Calibrate() {
  auto size = sizeof(RxGlobalHeader);
  auto body = std::make_unique<uint8_t[]>(size);

  auto *header = reinterpret_cast<RxGlobalHeader *>(&body[0]);
  header->msg_id = CMD_INIT_REF_CLOCK;
  header->command = CMD_INIT_REF_CLOCK;

  this->_link->Send(size, std::move(body));
  return this->WaitMsgProcessed(CMD_INIT_REF_CLOCK, 5000);
}

bool AUTDController::Clear() {
  auto size = sizeof(RxGlobalHeader);
  auto body = std::make_unique<uint8_t[]>(size);

  auto *header = reinterpret_cast<RxGlobalHeader *>(&body[0]);
  header->msg_id = CMD_CLEAR;
  header->command = CMD_CLEAR;

  this->_link->Send(size, std::move(body));
  return this->WaitMsgProcessed(CMD_CLEAR, 200);
}

void AUTDController::Close() {
  if (this->is_open()) {
    this->FinishSTModulation();
    this->Flush();
    this->Stop();
    this->Clear();
    this->_link->Close();
    this->_build_cond.notify_all();
    if (std::this_thread::get_id() != this->_build_thr.get_id() && this->_build_thr.joinable()) this->_build_thr.join();
    this->_send_cond.notify_all();
    if (std::this_thread::get_id() != this->_send_thr.get_id() && this->_send_thr.joinable()) this->_send_thr.join();
    this->_link = nullptr;
  }
}

void AUTDController::Stop() {
  auto nullgain = autd::gain::NullGain::Create();
  this->AppendGainSync(nullgain, true);
}

void AUTDController::AppendGain(GainPtr gain) {
  this->_p_stm_timer->Stop();
  gain->SetGeometry(this->_geometry);
  {
    std::unique_lock<std::mutex> lk(_build_mtx);
    _build_q.push(gain);
  }
  _build_cond.notify_all();
}

void AUTDController::AppendGainSync(GainPtr gain, bool wait_for_send) {
  this->_p_stm_timer->Stop();
  try {
    gain->SetGeometry(this->_geometry);
    if (!gain->built()) gain->Build();

    size_t body_size = 0;
    uint8_t msg_id = 0;
    auto body = this->MakeBody(gain, nullptr, &body_size, &msg_id);
    if (this->is_open()) this->_link->Send(body_size, move(body));
    if (wait_for_send) WaitMsgProcessed(msg_id);
  } catch (const int errnum) {
    this->_link->Close();
    std::cerr << errnum << "Link closed." << std::endl;
  }
}
void AUTDController::AppendModulation(ModulationPtr mod) {
  {
    std::unique_lock<std::mutex> lk(_send_mtx);
    _send_mod_q.push(mod);
  }
  _send_cond.notify_all();
}
void AUTDController::AppendModulationSync(ModulationPtr mod) {
  try {
    if (this->is_open()) {
      while (mod->buffer.size() > mod->_sent) {
        size_t body_size = 0;
        uint8_t msg_id = 0;
        auto body = this->MakeBody(nullptr, mod, &body_size, &msg_id);
        this->_link->Send(body_size, move(body));
        WaitMsgProcessed(msg_id);
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
  auto len = this->_stm_gains.size();
  auto itvl_us = static_cast<int>(1000000. / freq / len);
  this->_p_stm_timer->SetInterval(itvl_us);

  auto current_size = this->_stm_bodies.size();
  this->_stm_bodies.resize(len);
  this->_stm_body_sizes.resize(len);

  for (size_t i = current_size; i < len; i++) {
    auto g = this->_stm_gains[i];
    g->SetGeometry(this->_geometry);
    if (!g->built()) g->Build();

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
  std::vector<size_t>().swap(this->_stm_body_sizes);
}

void AUTDController::AppendSequence(SequencePtr seq) {
  while (seq->_sent < seq->control_points().size()) {
    size_t body_size = 0;
    uint8_t msg_id = 0;
    auto body = this->MakeSeqBody(seq, &body_size, &msg_id);
    this->_link->Send(body_size, move(body));
    if (seq->_sent == seq->control_points().size()) {
      this->WaitMsgProcessed(0xC0, 2000, 0xE0);
    } else {
      this->WaitMsgProcessed(msg_id, 200);
    }
  }
  this->CalibrateSeq();
}

void AUTDController::CalibrateSeq() {
  std::vector<uint16_t> laps;
  for (size_t i = 0; i < _rx_data.size() / 2; i++) {
    auto lap_raw = (static_cast<uint16_t>(_rx_data[2 * i + 1]) << 8) | _rx_data[2 * i];
    laps.push_back(lap_raw & 0x03FF);
  }

  std::vector<uint16_t> diffs;
  auto minimum = *std::min_element(laps.begin(), laps.end());
  for (size_t i = 0; i < laps.size(); i++) {
    diffs.push_back(laps[i] - minimum);
  }

  auto diff_max = *std::max_element(diffs.begin(), diffs.end());
  if (diff_max == 0) {
    return;
  } else if (diff_max > 500) {
    for (size_t i = 0; i < laps.size(); i++) {
      laps[i] = laps[i] < 500 ? laps[i] + 1000 : laps[i];
    }
    minimum = *std::min_element(laps.begin(), laps.end());
    for (size_t i = 0; i < laps.size(); i++) {
      diffs[i] = laps[i] - minimum;
    }
  }

  size_t body_size = 0;
  auto calib_body = this->MakeCalibBody(diffs, &body_size);
  this->_link->Send(body_size, std::move(calib_body));
  this->WaitMsgProcessed(0xE0, 200, 0xE0);
}

void AUTDController::Flush() {
  std::unique_lock<std::mutex> lk0(_send_mtx);
  std::unique_lock<std::mutex> lk1(_build_mtx);
  std::queue<GainPtr>().swap(_build_q);
  std::queue<GainPtr>().swap(_send_gain_q);
  std::queue<ModulationPtr>().swap(_send_mod_q);
}

bool AUTDController::WaitMsgProcessed(uint8_t msg_id, size_t max_trial, uint8_t mask) {
  auto success = false;
  auto num_dev = this->_geometry->numDevices();
  auto buffer_len = num_dev * EC_INPUT_FRAME_SIZE;
  for (size_t i = 0; i < max_trial; i++) {
    _rx_data = this->_link->Read(static_cast<uint32_t>(buffer_len));
    size_t processed = 0;
    for (size_t dev = 0; dev < num_dev; dev++) {
      uint8_t proc_id = _rx_data[dev * 2 + 1] & mask;
      if (proc_id == msg_id) processed++;
    }

    if (processed == num_dev) {
      return true;
    }

    auto wait = static_cast<size_t>(std::ceil(EC_TRAFFIC_DELAY * 1000 / EC_DEVICE_PER_FRAME * num_dev));
    std::this_thread::sleep_for(std::chrono::milliseconds(wait));
  }
  return false;
}

FirmwareInfoList AUTDController::firmware_info_list() {
  auto size = this->_geometry->numDevices();

  FirmwareInfoList res;
  auto make_header = [](uint8_t command) {
    auto header_bytes = std::make_unique<uint8_t[]>(sizeof(RxGlobalHeader));
    auto *header = reinterpret_cast<RxGlobalHeader *>(&header_bytes[0]);
    header->msg_id = command;
    header->command = command;
    return header_bytes;
  };

  std::vector<uint16_t> cpu_versions(size);
  std::vector<uint16_t> fpga_versions(size);

  std::unique_ptr<uint8_t[]> header;

  auto send_size = sizeof(RxGlobalHeader);
  header = make_header(CMD_READ_CPU_VER_LSB);
  this->_link->Send(send_size, std::move(header));
  WaitMsgProcessed(CMD_READ_CPU_VER_LSB, 50);
  for (size_t i = 0; i < size; i++) {
    cpu_versions[i] = _rx_data[2 * i];
  }
  header = make_header(CMD_READ_CPU_VER_MSB);
  this->_link->Send(send_size, std::move(header));
  WaitMsgProcessed(CMD_READ_CPU_VER_MSB, 50);
  for (size_t i = 0; i < size; i++) {
    cpu_versions[i] = ((uint16_t)_rx_data[2 * i] << 8) | cpu_versions[i];
  }

  header = make_header(CMD_READ_FPGA_VER_LSB);
  this->_link->Send(send_size, std::move(header));
  WaitMsgProcessed(CMD_READ_FPGA_VER_LSB, 50);
  for (size_t i = 0; i < size; i++) {
    fpga_versions[i] = _rx_data[2 * i];
  }

  header = make_header(CMD_READ_FPGA_VER_MSB);
  this->_link->Send(send_size, std::move(header));
  WaitMsgProcessed(CMD_READ_FPGA_VER_MSB, 50);
  for (size_t i = 0; i < size; i++) {
    fpga_versions[i] = ((uint16_t)_rx_data[2 * i] << 8) | fpga_versions[i];
  }

  for (uint16_t i = 0; i < size; i++) {
    FirmwareInfo info{i, cpu_versions[i], fpga_versions[i]};
    res.push_back(info);
  }
  return res;
}

void AUTDController::LateralModulationAT(Vector3 point, Vector3 dir, double lm_amp, double lm_freq) {
  auto p1 = point + lm_amp * dir;
  auto p2 = point - lm_amp * dir;
  this->FinishSTModulation();
  this->AppendSTMGain(autd::gain::FocalPointGain::Create(p1));
  this->AppendSTMGain(autd::gain::FocalPointGain::Create(p2));
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
        uint8_t msg_id = 0;
        auto body = MakeBody(gain, mod, &body_size, &msg_id);
        if (this->_link->is_open()) this->_link->Send(body_size, move(body));
        if (mod != nullptr) WaitMsgProcessed(msg_id);

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

std::unique_ptr<uint8_t[]> AUTDController::MakeBody(GainPtr gain, ModulationPtr mod, size_t *const size, uint8_t *const send_msg_id) {
  auto num_devices = (gain != nullptr) ? gain->geometry()->numDevices() : 0;

  *size = sizeof(RxGlobalHeader) + sizeof(uint16_t) * NUM_TRANS_IN_UNIT * num_devices;
  auto body = std::make_unique<uint8_t[]>(*size);

  auto *header = reinterpret_cast<RxGlobalHeader *>(&body[0]);
  *send_msg_id = get_id();
  header->msg_id = *send_msg_id;
  header->control_flags = 0;
  header->mod_size = 0;
  header->command = CMD_OP;

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

std::unique_ptr<uint8_t[]> AUTDController::MakeSeqBody(SequencePtr seq, size_t *const size, uint8_t *const send_msg_id) {
  auto num_devices = this->geometry()->numDevices();

  *size = sizeof(RxGlobalHeader) + sizeof(uint16_t) * NUM_TRANS_IN_UNIT * num_devices;
  auto body = std::make_unique<uint8_t[]>(*size);

  auto *header = reinterpret_cast<RxGlobalHeader *>(&body[0]);
  *send_msg_id = get_id();
  header->msg_id = *send_msg_id;
  header->control_flags = SEQ_MODE;
  header->command = CMD_SEQ_MODE;
  header->mod_size = 0;
  if (this->_silent_mode) header->control_flags |= SILENT;

  uint16_t send_size = std::max(0, std::min(static_cast<int>(seq->control_points().size() - seq->_sent), 40));
  header->seq_size = send_size;
  header->seq_div = seq->_sampl_freq_div;

  if (seq->_sent == 0) {
    header->control_flags |= SEQ_BEGIN;
  }
  if (seq->_sent + send_size >= seq->control_points().size()) {
    header->control_flags |= SEQ_END;
  }

  auto *cursor = &body[0] + sizeof(RxGlobalHeader) / sizeof(body[0]);
  for (int device = 0; device < num_devices; device++) {
    std::vector<uint8_t> foci;
    foci.reserve(static_cast<size_t>(send_size) * 10);

    for (int i = 0; i < send_size; i++) {
      auto v64 = this->geometry()->local_position(device, seq->control_points()[seq->_sent + i]);
      auto x = static_cast<uint32_t>(static_cast<int32_t>(v64.x() / FIXED_NUM_UNIT));
      auto y = static_cast<uint32_t>(static_cast<int32_t>(v64.y() / FIXED_NUM_UNIT));
      auto z = static_cast<uint32_t>(static_cast<int32_t>(v64.z() / FIXED_NUM_UNIT));
      foci.push_back(static_cast<uint8_t>(x & 0x000000FF));
      foci.push_back(static_cast<uint8_t>((x & 0x0000FF00) >> 8));
      foci.push_back(static_cast<uint8_t>(((x & 0x80000000) >> 24) | ((x & 0x007F0000) >> 16)));
      foci.push_back(static_cast<uint8_t>(y & 0x000000FF));
      foci.push_back(static_cast<uint8_t>((y & 0x0000FF00) >> 8));
      foci.push_back(static_cast<uint8_t>(((y & 0x80000000) >> 24) | ((y & 0x007F0000) >> 16)));
      foci.push_back(static_cast<uint8_t>(z & 0x000000FF));
      foci.push_back(static_cast<uint8_t>((z & 0x0000FF00) >> 8));
      foci.push_back(static_cast<uint8_t>(((z & 0x80000000) >> 24) | ((z & 0x007F0000) >> 16)));
      foci.push_back(0xFF);  // amp
    }
    std::memcpy(cursor, &foci[0], foci.size());
    cursor += NUM_TRANS_IN_UNIT * 2;
  }

  seq->_sent += send_size;
  return body;
}

std::unique_ptr<uint8_t[]> AUTDController::MakeCalibBody(std::vector<uint16_t> diffs, size_t *const size) {
  *size = sizeof(RxGlobalHeader) + sizeof(uint16_t) * NUM_TRANS_IN_UNIT * diffs.size();
  auto body = std::make_unique<uint8_t[]>(*size);

  auto *header = reinterpret_cast<RxGlobalHeader *>(&body[0]);
  header->msg_id = CMD_CALIB_SEQ_CLOCK;
  header->control_flags = 0;
  header->command = CMD_CALIB_SEQ_CLOCK;

  uint16_t *cursor = reinterpret_cast<uint16_t *>(&body[sizeof(RxGlobalHeader)]);
  for (size_t i = 0; i < diffs.size(); i++) {
    *cursor = diffs[i];
    cursor += NUM_TRANS_IN_UNIT;
  }

  return body;
}

ControllerPtr Controller::Create() { return std::make_shared<AUTDController>(); }
}  // namespace autd
