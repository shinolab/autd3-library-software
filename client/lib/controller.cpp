// File: controller.cpp
// Project: lib
// Created Date: 13/05/2016
// Author: Seki Inoue
// -----
// Last Modified: 21/12/2020
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

ControllerPtr Controller::Create() { return std::make_shared<AUTDController>(); }

AUTDController::AUTDController() {
  this->_link = nullptr;
  this->_geometry = Geometry::Create();
  this->_silent_mode = true;
  this->_p_stm_timer = std::make_unique<Timer>();
  this->_seq_mode = false;
}

AUTDController::~AUTDController() {
  if (std::this_thread::get_id() != this->_build_gain_thr.get_id() && this->_build_gain_thr.joinable()) this->_build_gain_thr.join();
  if (std::this_thread::get_id() != this->_build_mod_thr.get_id() && this->_build_mod_thr.joinable()) this->_build_mod_thr.join();
  if (std::this_thread::get_id() != this->_send_thr.get_id() && this->_send_thr.joinable()) this->_send_thr.join();
}

bool AUTDController::is_open() { return this->_link != nullptr && this->_link->is_open(); }

GeometryPtr AUTDController::geometry() noexcept { return this->_geometry; }

bool AUTDController::silent_mode() noexcept { return this->_silent_mode; }

size_t AUTDController::remaining_in_buffer() {
  return this->_send_gain_q.size() + this->_send_mod_q.size() + this->_build_gain_q.size() + this->_build_mod_q.size();
}

void AUTDController::SetSilentMode(bool silent) noexcept { this->_silent_mode = silent; }

void AUTDController::OpenWith(LinkPtr link) {
  this->Close();

  this->_link = link;
  this->_link->Open();
  if (this->_link->is_open())
    this->InitPipeline();
  else
    this->Close();
}

void AUTDController::CloseLink() { this->_link->Close(); }

size_t &AUTDController::mod_sent(ModulationPtr mod) { return mod->_sent; }
size_t &AUTDController::seq_sent(SequencePtr seq) { return seq->_sent; }
uint16_t AUTDController::seq_div(SequencePtr seq) { return seq->_sampl_freq_div; }

const uint16_t *AUTDController::gain_data_addr(GainPtr gain, int device_id) { return &gain->_data[device_id].at(0); }

void AUTDController::Stop() {
  auto nullgain = autd::gain::NullGain::Create();
  this->AppendGainSync(nullgain, true);
}

void AUTDController::AppendModulation(ModulationPtr mod) {
  {
    std::unique_lock<std::mutex> lk(_build_mod_mtx);
    _build_mod_q.push(mod);
  }
  _build_mod_cond.notify_all();
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

void AUTDController::Flush() {
  std::unique_lock<std::mutex> lk0(_send_mtx);
  std::unique_lock<std::mutex> lk1(_build_gain_mtx);
  std::unique_lock<std::mutex> lk2(_build_mod_mtx);
  std::queue<GainPtr>().swap(_build_gain_q);
  std::queue<ModulationPtr>().swap(_build_mod_q);
  std::queue<GainPtr>().swap(_send_gain_q);
  std::queue<ModulationPtr>().swap(_send_mod_q);
}

void AUTDController::LateralModulationAT(Vector3 point, Vector3 dir, double lm_amp, double lm_freq) {
  auto p1 = point + lm_amp * dir;
  auto p2 = point - lm_amp * dir;
  this->FinishSTModulation();
  this->AppendSTMGain(autd::gain::FocalPointGain::Create(p1));
  this->AppendSTMGain(autd::gain::FocalPointGain::Create(p2));
  this->StartSTModulation(lm_freq);
}

bool AUTDController::Clear() {
  this->_config = Configuration::GetDefaultConfiguration();

  auto size = sizeof(RxGlobalHeaderV_0_6);
  auto body = std::make_unique<uint8_t[]>(size);

  auto *header = reinterpret_cast<RxGlobalHeaderV_0_6 *>(&body[0]);
  header->msg_id = CMD_CLEAR;
  header->command = CMD_CLEAR;

  this->SendData(size, std::move(body));
  return this->WaitMsgProcessed(CMD_CLEAR, 200);
}

void AUTDController::Close() {
  if (this->is_open()) {
    this->FinishSTModulation();
    this->Flush();
    this->Stop();
    this->Clear();
    this->CloseLink();
    this->_build_gain_cond.notify_all();
    if (std::this_thread::get_id() != this->_build_gain_thr.get_id() && this->_build_gain_thr.joinable()) this->_build_gain_thr.join();
    this->_build_mod_cond.notify_all();
    if (std::this_thread::get_id() != this->_build_mod_thr.get_id() && this->_build_mod_thr.joinable()) this->_build_mod_thr.join();
    this->_send_cond.notify_all();
    if (std::this_thread::get_id() != this->_send_thr.get_id() && this->_send_thr.joinable()) this->_send_thr.join();
    this->_link = nullptr;
  }
}

void AUTDController::AppendGain(GainPtr gain) {
  this->_p_stm_timer->Stop();
  this->_seq_mode = false;
  gain->SetGeometry(this->_geometry);
  {
    std::unique_lock<std::mutex> lk(_build_gain_mtx);
    _build_gain_q.push(gain);
  }
  _build_gain_cond.notify_all();
}

void AUTDController::AppendGainSync(GainPtr gain, bool wait_for_send) {
  this->_p_stm_timer->Stop();
  this->_seq_mode = false;
  try {
    gain->SetGeometry(this->_geometry);
    gain->Build();
    this->_link_manager->SendBlocking(gain, nullptr);
  } catch (const int errnum) {
    this->CloseLink();
    std::cerr << errnum << "Link closed." << std::endl;
  }
}

void AUTDController::AppendModulationSync(ModulationPtr mod) {
  try {
    mod->Build(this->_config);
    if (this->is_open()) {
      while (mod->buffer.size() > this->mod_sent(mod)) {
        size_t body_size = 0;
        uint8_t msg_id = 0;
        auto body = this->MakeBody(nullptr, mod, &body_size, &msg_id);
        this->SendData(body_size, move(body));
        WaitMsgProcessed(msg_id);
      }
      this->mod_sent(mod) = 0;
    }
  } catch (const int errnum) {
    this->Close();
    std::cerr << errnum << "Link closed." << std::endl;
  }
}

void AUTDController::AppendSequence(SequencePtr seq) {
  this->_seq_mode = true;
  while (this->seq_sent(seq) < seq->control_points().size()) {
    size_t body_size = 0;
    uint8_t msg_id = 0;
    auto body = this->MakeSeqBody(seq, &body_size, &msg_id);
    this->SendData(body_size, move(body));
    if (this->seq_sent(seq) == seq->control_points().size()) {
      this->WaitMsgProcessed(0xC0, 2000, 0xE0);
    } else {
      this->WaitMsgProcessed(msg_id, 200);
    }
  }
  this->CalibrateSeq();
}

void AUTDController::CalibrateSeq() {
  std::vector<uint16_t> laps;
  for (size_t i = 0; i < this->_rx_data.size() / 2; i++) {
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
  this->SendData(body_size, std::move(calib_body));
  this->WaitMsgProcessed(0xE0, 200, 0xE0);
}

FirmwareInfoList AUTDController::firmware_info_list() {
  auto size = this->_geometry->numDevices();

  FirmwareInfoList res;
  auto make_header = [](uint8_t command) {
    auto header_bytes = std::make_unique<uint8_t[]>(sizeof(RxGlobalHeaderV_0_6));
    auto *header = reinterpret_cast<RxGlobalHeaderV_0_6 *>(&header_bytes[0]);
    header->msg_id = command;
    header->command = command;
    return header_bytes;
  };

  std::vector<uint16_t> cpu_versions(size);
  std::vector<uint16_t> fpga_versions(size);

  std::unique_ptr<uint8_t[]> header;

  auto send_size = sizeof(RxGlobalHeaderV_0_6);
  header = make_header(CMD_READ_CPU_VER_LSB);
  this->SendData(send_size, std::move(header));
  WaitMsgProcessed(CMD_READ_CPU_VER_LSB, 50);
  for (size_t i = 0; i < size; i++) {
    cpu_versions[i] = _rx_data[2 * i];
  }
  header = make_header(CMD_READ_CPU_VER_MSB);
  this->SendData(send_size, std::move(header));
  WaitMsgProcessed(CMD_READ_CPU_VER_MSB, 50);
  for (size_t i = 0; i < size; i++) {
    cpu_versions[i] = ((uint16_t)_rx_data[2 * i] << 8) | cpu_versions[i];
  }

  header = make_header(CMD_READ_FPGA_VER_LSB);
  this->SendData(send_size, std::move(header));
  WaitMsgProcessed(CMD_READ_FPGA_VER_LSB, 50);
  for (size_t i = 0; i < size; i++) {
    fpga_versions[i] = _rx_data[2 * i];
  }

  header = make_header(CMD_READ_FPGA_VER_MSB);
  this->SendData(send_size, std::move(header));
  WaitMsgProcessed(CMD_READ_FPGA_VER_MSB, 50);
  for (size_t i = 0; i < size; i++) {
    fpga_versions[i] = ((uint16_t)_rx_data[2 * i] << 8) | fpga_versions[i];
  }

  for (auto i = 0; i < size; i++) {
    auto info = AUTDController::FirmwareInfoCreate(static_cast<uint16_t>(i), cpu_versions[i], fpga_versions[i]);
    res.push_back(info);
  }
  return res;
}

void AUTDController::InitPipeline() {
  this->_build_gain_thr = std::thread([&] {
    while (this->is_open()) {
      GainPtr gain = nullptr;
      {
        std::unique_lock<std::mutex> lk(_build_gain_mtx);

        _build_gain_cond.wait(lk, [&] { return _build_gain_q.size() || !this->is_open(); });

        if (_build_gain_q.size() > 0) {
          gain = _build_gain_q.front();
          _build_gain_q.pop();
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

  this->_build_mod_thr = std::thread([&] {
    while (this->is_open()) {
      ModulationPtr mod = nullptr;
      {
        std::unique_lock<std::mutex> lk(_build_mod_mtx);

        _build_mod_cond.wait(lk, [&] { return _build_mod_q.size() || !this->is_open(); });

        if (_build_mod_q.size() > 0) {
          mod = _build_mod_q.front();
          _build_mod_q.pop();
        }
      }

      if (mod != nullptr) {
        mod->Build(_config);
        {
          std::unique_lock<std::mutex> lk(_send_mtx);
          _send_mod_q.push(mod);
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
        if (mod != nullptr) WaitMsgProcessed(msg_id);

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

std::unique_ptr<uint8_t[]> AUTDController::MakeSeqBody(SequencePtr seq, size_t *const size, uint8_t *const send_msg_id) {
  auto num_devices = this->geometry()->numDevices();

  *size = sizeof(RxGlobalHeaderV_0_6) + sizeof(uint16_t) * NUM_TRANS_IN_UNIT * num_devices;
  auto body = std::make_unique<uint8_t[]>(*size);

  auto *header = reinterpret_cast<RxGlobalHeaderV_0_6 *>(&body[0]);
  *send_msg_id = get_id();
  header->msg_id = *send_msg_id;
  header->control_flags = SEQ_MODE;
  header->command = CMD_SEQ_MODE;
  header->mod_size = 0;
  if (this->_silent_mode) header->control_flags |= SILENT;

  uint16_t send_size = std::max(0, std::min(static_cast<int>(seq->control_points().size() - this->seq_sent(seq)), 40));
  header->seq_size = send_size;
  header->seq_div = this->seq_div(seq);

  if (this->seq_sent(seq) == 0) {
    header->control_flags |= SEQ_BEGIN;
  }
  if (this->seq_sent(seq) + send_size >= seq->control_points().size()) {
    header->control_flags |= SEQ_END;
  }

  auto *cursor = &body[0] + sizeof(RxGlobalHeaderV_0_6) / sizeof(body[0]);
  for (int device = 0; device < num_devices; device++) {
    std::vector<uint8_t> foci;
    foci.reserve(static_cast<size_t>(send_size) * 10);

    for (int i = 0; i < send_size; i++) {
      auto v64 = this->geometry()->local_position(device, seq->control_points()[this->seq_sent(seq) + i]);
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

  this->seq_sent(seq) += send_size;
  return body;
}

std::unique_ptr<uint8_t[]> AUTDController::MakeCalibBody(std::vector<uint16_t> diffs, size_t *const size) {
  *size = sizeof(RxGlobalHeaderV_0_6) + sizeof(uint16_t) * NUM_TRANS_IN_UNIT * diffs.size();
  auto body = std::make_unique<uint8_t[]>(*size);

  auto *header = reinterpret_cast<RxGlobalHeaderV_0_6 *>(&body[0]);
  header->msg_id = CMD_CALIB_SEQ_CLOCK;
  header->control_flags = 0;
  header->command = CMD_CALIB_SEQ_CLOCK;

  uint16_t *cursor = reinterpret_cast<uint16_t *>(&body[sizeof(RxGlobalHeaderV_0_6)]);
  for (size_t i = 0; i < diffs.size(); i++) {
    *cursor = diffs[i];
    cursor += NUM_TRANS_IN_UNIT;
  }

  return body;
}

static inline uint16_t log2u(const uint32_t x) {
#ifdef _MSC_VER
  unsigned long n;  // NOLINT
  _BitScanReverse(&n, x);
#else
  uint32_t n;
  n = 31 - __builtin_clz(x);
#endif
  return static_cast<uint16_t>(n);
}

bool AUTDController::Calibrate(Configuration config) {
  this->_config = config;

  auto num_devices = this->_geometry->numDevices();
  auto size = sizeof(RxGlobalHeaderV_0_6) + sizeof(uint16_t) * NUM_TRANS_IN_UNIT * num_devices;
  auto body = std::make_unique<uint8_t[]>(size);

  auto *header = reinterpret_cast<RxGlobalHeaderV_0_6 *>(&body[0]);
  header->msg_id = CMD_INIT_REF_CLOCK;
  header->command = CMD_INIT_REF_CLOCK;

  auto mod_smpl_freq = static_cast<uint32_t>(_config.mod_sampling_freq());
  auto mod_buf_size = static_cast<uint32_t>(_config.mod_buf_size());

  if (mod_buf_size < mod_smpl_freq) {
    std::cerr << "Modulation buffer size must be not less than sampling frequency.\n";
    std::cerr << "Modulation buffer size is set to " << mod_smpl_freq << std::endl;
    this->_config.set_mod_buf_size(static_cast<MOD_BUF_SIZE>(mod_smpl_freq));
  }

  auto mod_idx_shift = log2u(MOD_SAMPLING_FREQ_BASE / mod_smpl_freq);
  auto ref_clk_cyc_shift = log2u(mod_buf_size / mod_smpl_freq);

  auto *cursor = reinterpret_cast<uint16_t *>(&body[0] + sizeof(RxGlobalHeaderV_0_6) / sizeof(body[0]));
  for (int i = 0; i < this->_geometry->numDevices(); i++) {
    cursor[0] = mod_idx_shift;
    cursor[1] = ref_clk_cyc_shift;
    cursor += NUM_TRANS_IN_UNIT;
  }

  this->SendData(size, std::move(body));
  return this->WaitMsgProcessed(CMD_INIT_REF_CLOCK, 5000);
}

namespace _internal {
using std::move;
using std::unique_ptr;
using std::vector;

class AUTDLinkManager {
 public:
  AUTDLinkManager(LinkPtr link) { _link = link; }
  ~AUTDLinkManager() {}

  void Send(GainPtr gain, ModulationPtr mod) {
    size_t body_size = 0;
    uint8_t msg_id = 0;
    auto body = this->MakeBody(gain, mod, &body_size, &msg_id);
    if (this->_link->is_open()) this->SendData(body_size, move(body));
  }

  void SendBlocking(GainPtr gain, ModulationPtr mod) {
    size_t body_size = 0;
    uint8_t msg_id = 0;
    auto body = this->MakeBody(gain, mod, &body_size, &msg_id);
    if (this->_link->is_open()) this->SendData(body_size, move(body));
    WaitMsgProcessed(msg_id);
  }

  void SendBlocking(SequencePtr seq) {
    size_t body_size = 0;
    uint8_t msg_id = 0;
    auto body = this->MakeSeqBody(seq, &body_size, &msg_id);
    this->SendData(body_size, move(body));
    if (seq.sent == seq->control_points().size()) {
      this->WaitMsgProcessed(0xC0, 2000, 0xE0);
    } else {
      this->WaitMsgProcessed(msg_id, 200);
    }
  }

 private:
  static uint8_t get_id() {
    static std::atomic<uint8_t> id{OP_MODE_MSG_ID_MIN - 1};

    id.fetch_add(0x01);
    uint8_t expected = OP_MODE_MSG_ID_MAX + 1;
    id.compare_exchange_weak(expected, OP_MODE_MSG_ID_MIN);

    return id.load();
  }

  void SendData(size_t size, unique_ptr<uint8_t[]> buf) { this->_link->Send(size, move(buf)); }
  vector<uint8_t> ReadData(uint32_t buffer_len) { return this->_link->Read(buffer_len); }

  bool WaitMsgProcessed(uint8_t msg_id, size_t max_trial = 200, uint8_t mask = 0xFF) {
    auto success = false;
    auto num_dev = this->_geometry->numDevices();
    auto buffer_len = num_dev * EC_INPUT_FRAME_SIZE;
    for (size_t i = 0; i < max_trial; i++) {
      _rx_data = this->ReadData(static_cast<uint32_t>(buffer_len));
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

  unique_ptr<uint8_t[]> MakeBody(GainPtr gain, ModulationPtr mod, size_t *const size, uint8_t *const send_msg_id) {
    auto num_devices = (gain != nullptr) ? gain->geometry()->numDevices() : 0;

    *size = sizeof(RxGlobalHeader) + sizeof(uint16_t) * NUM_TRANS_IN_UNIT * num_devices;
    auto body = std::make_unique<uint8_t[]>(*size);

    auto *header = reinterpret_cast<RxGlobalHeader *>(&body[0]);
    *send_msg_id = get_id();
    header->msg_id = *send_msg_id;
    header->control_flags = 0;
    header->mod_size = 0;
    header->command = CMD_OP;

    if (this->_seq_mode) header->control_flags |= SEQ_MODE;
    if (this->_silent_mode) header->control_flags |= SILENT;

    if (mod != nullptr) {
      const uint8_t mod_size = std::max(0, std::min(static_cast<int>(mod->buffer.size() - mod->sent), MOD_FRAME_SIZE));
      header->mod_size = mod_size;
      if (mod->sent == 0) header->control_flags |= LOOP_BEGIN;
      if (mod->sent + mod_size >= mod->buffer.size()) header->control_flags |= LOOP_END;

      std::memcpy(header->mod, &mod->buffer[this->mod_sent(mod)], mod_size);
      this->mod_sent(mod) += mod_size;
    }

    auto *cursor = &body[0] + sizeof(RxGlobalHeader) / sizeof(body[0]);
    auto byteSize = NUM_TRANS_IN_UNIT * sizeof(uint16_t);
    if (gain != nullptr) {
      for (int i = 0; i < gain->geometry()->numDevices(); i++) {
        auto deviceId = gain->geometry()->deviceIdForDeviceIdx(i);
        std::memcpy(cursor, this->gain_data_addr(gain, deviceId), byteSize);
        cursor += byteSize / sizeof(body[0]);
      }
    }
    return body;
  }

  unique_ptr<uint8_t[]> MakeSeqBody(SequencePtr seq, size_t *const size, uint8_t *const send_msg_id);
  void CalibrateSeq();
  unique_ptr<uint8_t[]> MakeCalibBody(vector<uint16_t> diffs, size_t *const size);

  LinkPtr _link;
};

}  // namespace _internal
}  // namespace autd
