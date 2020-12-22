// File: autd_logic.cpp
// Project: lib
// Created Date: 22/12/2020
// Author: Shun Suzuki
// -----
// Last Modified: 22/12/2020
// Modified By: Shun Suzuki (suzuki@hapis.k.u-tokyo.ac.jp)
// -----
// Copyright (c) 2020 Hapis Lab. All rights reserved.
//

#include "autd_logic.hpp"

namespace autd::_internal {

using std::move;

AUTDLogic::AUTDLogic() {
  this->_geometry = Geometry::Create();
  this->_link = nullptr;

  this->_silent_mode = true;
  this->_seq_mode = false;

  this->_config = Configuration::GetDefaultConfiguration();
}

bool AUTDLogic::is_open() { return (this->_link != nullptr) && this->_link->is_open(); }

void AUTDLogic::OpenWith(LinkPtr link) {
  this->_link = link;
  this->_link->Open();
}

void AUTDLogic::Send(GainPtr gain, ModulationPtr mod) {
  size_t body_size = 0;
  uint8_t msg_id = 0;
  auto body = this->MakeBody(gain, mod, &body_size, &msg_id);
  this->SendData(body_size, move(body));
}

void AUTDLogic::SendBlocking(GainPtr gain, ModulationPtr mod) {
  size_t body_size = 0;
  uint8_t msg_id = 0;
  auto body = this->MakeBody(gain, mod, &body_size, &msg_id);
  this->SendData(body_size, move(body));
  WaitMsgProcessed(msg_id);
}

void AUTDLogic::SendBlocking(SequencePtr seq) {
  size_t body_size = 0;
  uint8_t msg_id = 0;
  auto body = this->MakeBody(seq, &body_size, &msg_id);
  this->SendData(body_size, move(body));
  if (seq->_sent == seq->control_points().size()) {
    this->WaitMsgProcessed(0xC0, 2000, 0xE0);
  } else {
    this->WaitMsgProcessed(msg_id, 200);
  }
}

bool AUTDLogic::SendBlocking(size_t size, unique_ptr<uint8_t[]> data, size_t trial) {
  uint8_t msg_id = data[0];
  this->SendData(size, move(data));
  return this->WaitMsgProcessed(msg_id, trial, 0xFF);
}

void AUTDLogic::SendData(size_t size, unique_ptr<uint8_t[]> data) {
  if (this->_link == nullptr || !this->_link->is_open()) {
    std::cerr << "Link is closed." << std::endl;
  }

  try {
    this->_link->Send(size, move(data));
  } catch (const int errnum) {
    this->_link->Close();
    this->_link = nullptr;
    std::cerr << errnum << "Link closed." << std::endl;
  }
}

bool AUTDLogic::WaitMsgProcessed(uint8_t msg_id, size_t max_trial, uint8_t mask) {
  if (this->_link == nullptr || !this->_link->is_open()) {
    return false;
  }

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

bool AUTDLogic::Calibrate(Configuration config) {
  this->_config = config;
  size_t size = 0;
  auto body = this->MakeCalibBody(config, &size);
  return this->SendBlocking(size, std::move(body), 5000);
}

void AUTDLogic::CalibrateSeq() {
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
  auto calib_body = this->MakeCalibSeqBody(diffs, &body_size);
  this->SendData(body_size, std::move(calib_body));
  this->WaitMsgProcessed(0xE0, 200, 0xE0);
}

bool AUTDLogic::Clear() {
  this->_config = Configuration::GetDefaultConfiguration();

  auto size = sizeof(RxGlobalHeader);
  auto body = std::make_unique<uint8_t[]>(size);

  auto *header = reinterpret_cast<RxGlobalHeader *>(&body[0]);
  header->msg_id = CMD_CLEAR;
  header->command = CMD_CLEAR;

  return this->SendBlocking(size, std::move(body), 200);
}

void AUTDLogic::Close() {
  this->Clear();
  this->_link->Close();
  this->_link = nullptr;
}

FirmwareInfoList AUTDLogic::firmware_info_list() {
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
    auto info = FirmwareInfo(static_cast<uint16_t>(i), cpu_versions[i], fpga_versions[i]);
    res.push_back(info);
  }
  return res;
}

unique_ptr<uint8_t[]> AUTDLogic::MakeBody(GainPtr gain, ModulationPtr mod, size_t *const size, uint8_t *const send_msg_id) {
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
    const uint8_t mod_size = std::max(0, std::min(static_cast<int>(mod->buffer.size() - mod->_sent), MOD_FRAME_SIZE));
    header->mod_size = mod_size;
    if (mod->_sent == 0) header->control_flags |= LOOP_BEGIN;
    if (mod->_sent + mod_size >= mod->buffer.size()) header->control_flags |= LOOP_END;

    std::memcpy(header->mod, &mod->buffer[mod->_sent], mod_size);
    mod->_sent += mod_size;
  }

  auto *cursor = &body[0] + sizeof(RxGlobalHeader) / sizeof(body[0]);
  auto byteSize = NUM_TRANS_IN_UNIT * sizeof(uint16_t);
  if (gain != nullptr) {
    for (int i = 0; i < gain->geometry()->numDevices(); i++) {
      auto deviceId = gain->geometry()->deviceIdForDeviceIdx(i);
      std::memcpy(cursor, &gain->_data[deviceId].at(0), byteSize);
      cursor += byteSize / sizeof(body[0]);
    }
  }
  return body;
}

unique_ptr<uint8_t[]> AUTDLogic::MakeBody(SequencePtr seq, size_t *const size, uint8_t *const send_msg_id) {
  auto num_devices = this->_geometry->numDevices();

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
      auto v64 = this->_geometry->local_position(device, seq->control_points()[seq->_sent + i]);
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

unique_ptr<uint8_t[]> AUTDLogic::MakeCalibBody(Configuration config, size_t *const size) {
  this->_config = config;

  auto num_devices = this->_geometry->numDevices();
  *size = sizeof(RxGlobalHeader) + sizeof(uint16_t) * NUM_TRANS_IN_UNIT * num_devices;
  auto body = std::make_unique<uint8_t[]>(*size);

  auto *header = reinterpret_cast<RxGlobalHeader *>(&body[0]);
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

  auto *cursor = reinterpret_cast<uint16_t *>(&body[0] + sizeof(RxGlobalHeader) / sizeof(body[0]));
  for (int i = 0; i < num_devices; i++) {
    cursor[0] = mod_idx_shift;
    cursor[1] = ref_clk_cyc_shift;
    cursor += NUM_TRANS_IN_UNIT;
  }

  return body;
}

unique_ptr<uint8_t[]> AUTDLogic::MakeCalibSeqBody(std::vector<uint16_t> comps, size_t *const size) {
  *size = sizeof(RxGlobalHeader) + sizeof(uint16_t) * NUM_TRANS_IN_UNIT * comps.size();
  auto body = std::make_unique<uint8_t[]>(*size);

  auto *header = reinterpret_cast<RxGlobalHeader *>(&body[0]);
  header->msg_id = CMD_CALIB_SEQ_CLOCK;
  header->control_flags = 0;
  header->command = CMD_CALIB_SEQ_CLOCK;

  uint16_t *cursor = reinterpret_cast<uint16_t *>(&body[sizeof(RxGlobalHeader)]);
  for (size_t i = 0; i < comps.size(); i++) {
    *cursor = comps[i];
    cursor += NUM_TRANS_IN_UNIT;
  }

  return body;
}
}  // namespace autd::_internal
