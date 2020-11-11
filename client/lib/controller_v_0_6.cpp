// File: controller.cpp
// Project: lib
// Created Date: 13/05/2016
// Author: Seki Inoue
// -----
// Last Modified: 10/11/2020
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

#include "configuration.hpp"
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

AUTDControllerV_0_6::AUTDControllerV_0_6() : AUTDControllerV_0_1() { this->_seq_mode = false; }

AUTDControllerV_0_6::~AUTDControllerV_0_6() {}

bool AUTDControllerV_0_6::Calibrate(Configuration config) {
  if ((config.mod_sampling_freq() != MOD_SAMPLING_FREQ::SMPL_4_KHZ) || (config.mod_buf_size() != MOD_BUF_SIZE::BUF_4000)) {
    std::cerr << "Configurations are not available in this version." << std::endl;
  }

  auto size = sizeof(RxGlobalHeaderV_0_6);
  auto body = std::make_unique<uint8_t[]>(size);

  auto *header = reinterpret_cast<RxGlobalHeaderV_0_6 *>(&body[0]);
  header->msg_id = CMD_INIT_REF_CLOCK;
  header->command = CMD_INIT_REF_CLOCK;

  this->SendData(size, std::move(body));
  return this->WaitMsgProcessed(CMD_INIT_REF_CLOCK, 5000);
}

bool AUTDControllerV_0_6::Clear() {
  this->_config = Configuration::GetDefaultConfiguration();

  auto size = sizeof(RxGlobalHeaderV_0_6);
  auto body = std::make_unique<uint8_t[]>(size);

  auto *header = reinterpret_cast<RxGlobalHeaderV_0_6 *>(&body[0]);
  header->msg_id = CMD_CLEAR;
  header->command = CMD_CLEAR;

  this->SendData(size, std::move(body));
  return this->WaitMsgProcessed(CMD_CLEAR, 200);
}

void AUTDControllerV_0_6::Close() {
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

void AUTDControllerV_0_6::AppendGain(GainPtr gain) {
  this->_p_stm_timer->Stop();
  this->_seq_mode = false;
  gain->SetGeometry(this->_geometry);
  {
    std::unique_lock<std::mutex> lk(_build_gain_mtx);
    _build_gain_q.push(gain);
  }
  _build_gain_cond.notify_all();
}

void AUTDControllerV_0_6::AppendGainSync(GainPtr gain, bool wait_for_send) {
  this->_p_stm_timer->Stop();
  this->_seq_mode = false;
  try {
    gain->SetGeometry(this->_geometry);
    gain->Build();

    size_t body_size = 0;
    uint8_t msg_id = 0;
    auto body = this->MakeBody(gain, nullptr, &body_size, &msg_id);
    if (this->is_open()) this->SendData(body_size, move(body));
    if (wait_for_send) WaitMsgProcessed(msg_id);
  } catch (const int errnum) {
    this->CloseLink();
    std::cerr << errnum << "Link closed." << std::endl;
  }
}

void AUTDControllerV_0_6::AppendModulationSync(ModulationPtr mod) {
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

void AUTDControllerV_0_6::AppendSequence(SequencePtr seq) {
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

void AUTDControllerV_0_6::CalibrateSeq() {
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

bool AUTDControllerV_0_6::WaitMsgProcessed(uint8_t msg_id, size_t max_trial, uint8_t mask) {
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

FirmwareInfoList AUTDControllerV_0_6::firmware_info_list() {
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

void AUTDControllerV_0_6::InitPipeline() {
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

std::unique_ptr<uint8_t[]> AUTDControllerV_0_6::MakeBody(GainPtr gain, ModulationPtr mod, size_t *const size, uint8_t *const send_msg_id) {
  auto num_devices = (gain != nullptr) ? gain->geometry()->numDevices() : 0;

  *size = sizeof(RxGlobalHeaderV_0_6) + sizeof(uint16_t) * NUM_TRANS_IN_UNIT * num_devices;
  auto body = std::make_unique<uint8_t[]>(*size);

  auto *header = reinterpret_cast<RxGlobalHeaderV_0_6 *>(&body[0]);
  *send_msg_id = get_id();
  header->msg_id = *send_msg_id;
  header->control_flags = 0;
  header->mod_size = 0;
  header->command = CMD_OP;

  if (this->_seq_mode) header->control_flags |= SEQ_MODE;
  if (this->_silent_mode) header->control_flags |= SILENT;

  if (mod != nullptr) {
    const uint8_t mod_size = std::max(0, std::min(static_cast<int>(mod->buffer.size() - this->mod_sent(mod)), MOD_FRAME_SIZE_V_0_6));
    header->mod_size = mod_size;
    if (this->mod_sent(mod) == 0) header->control_flags |= LOOP_BEGIN;
    if (this->mod_sent(mod) + mod_size >= mod->buffer.size()) header->control_flags |= LOOP_END;

    std::memcpy(header->mod, &mod->buffer[this->mod_sent(mod)], mod_size);
    this->mod_sent(mod) += mod_size;
  }

  auto *cursor = &body[0] + sizeof(RxGlobalHeaderV_0_6) / sizeof(body[0]);
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

std::unique_ptr<uint8_t[]> AUTDControllerV_0_6::MakeSeqBody(SequencePtr seq, size_t *const size, uint8_t *const send_msg_id) {
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

std::unique_ptr<uint8_t[]> AUTDControllerV_0_6::MakeCalibBody(std::vector<uint16_t> diffs, size_t *const size) {
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
}  // namespace _internal
}  // namespace autd
