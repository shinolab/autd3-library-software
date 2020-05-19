// File: soem_link.cpp
// Project: lib
// Created Date: 24/08/2019
// Author: Shun Suzuki
// -----
// Last Modified: 19/05/2020
// Modified By: Shun Suzuki (suzuki@hapis.k.u-tokyo.ac.jp)
// -----
// Copyright (c) 2019-2020 Hapis Lab. All rights reserved.
//

#include "soem_link.hpp"

#include <algorithm>
#include <bitset>
#include <chrono>
#include <cstring>
#include <iostream>
#include <memory>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "autdsoem.hpp"
#include "ec_config.hpp"
#include "privdef.hpp"

namespace autd {

EtherCATAdapters SOEMLink::EnumerateAdapters(int *const size) {
  auto adapters = autdsoem::EtherCATAdapterInfo::EnumerateAdapters();
  *size = static_cast<int>(adapters.size());
#if DLL_FOR_CAPI
  EtherCATAdapters res = new EtherCATAdapter[*size];
  int i = 0;
#else
  EtherCATAdapters res;
#endif
  for (auto adapter : autdsoem::EtherCATAdapterInfo::EnumerateAdapters()) {
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

class SOEMLinkImpl : public SOEMLink {
 public:
  ~SOEMLinkImpl() override {}

  std::vector<uint8_t> WaitProcMsg(uint8_t id, uint8_t mask);
  std::unique_ptr<autdsoem::ISOEMController> _cnt;
  size_t _device_num = 0;
  std::string _ifname;
  autdsoem::ECConfig _config{};

 protected:
  void Open() final;
  void Close() final;
  void Send(size_t size, std::unique_ptr<uint8_t[]> buf) final;
  std::vector<uint8_t> Read(uint32_t buffer_len) final;
  bool is_open() final;
  bool CalibrateModulation() final;
};

LinkPtr SOEMLink::Create(std::string ifname, int device_num) {
  auto link = CreateHelper<SOEMLinkImpl>();
  link->_ifname = ifname;
  link->_device_num = device_num;

  return link;
}

void SOEMLinkImpl::Open() {
  _cnt = autdsoem::ISOEMController::Create();

  _config = autdsoem::ECConfig{};
  _config.ec_sm3_cyctime_ns = EC_SM3_CYCLE_TIME_NANO_SEC;
  _config.ec_sync0_cyctime_ns = EC_SYNC0_CYCLE_TIME_NANO_SEC;
  _config.header_size = HEADER_SIZE;
  _config.body_size = NUM_TRANS_IN_UNIT * 2;
  _config.input_frame_size = EC_INPUT_FRAME_SIZE;

  _cnt->Open(_ifname.c_str(), _device_num, _config);
}

void SOEMLinkImpl::Close() {
  if (_cnt->is_open()) {
    _cnt->Close();
  }
}

void SOEMLinkImpl::Send(size_t size, std::unique_ptr<uint8_t[]> buf) {
  if (_cnt->is_open()) {
    _cnt->Send(size, std::move(buf));
  }
}

std::vector<uint8_t> SOEMLinkImpl::Read(uint32_t _buffer_len) { return _cnt->Read(); }

bool SOEMLinkImpl::is_open() { return _cnt->is_open(); }

std::vector<uint8_t> SOEMLinkImpl::WaitProcMsg(uint8_t id, uint8_t mask) {
  std::vector<uint8_t> rx;
  for (size_t i = 0; i < 200; i++) {
    rx = Read(0);
    size_t processed = 0;
    for (size_t dev = 0; dev < _device_num; dev++) {
      uint8_t proc_id = rx[dev * 2 + 1] & mask;
      if (proc_id == id) processed++;
    }

    if (processed == _device_num) return rx;

    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
  return rx;
}

bool SOEMLinkImpl::CalibrateModulation() {
  if (_device_num == 0 || _device_num == 1) {
    return true;
  }

  struct CalibrationStatus {
    bool danger;
    uint16_t sync_base;
  };

  auto make_calib_header = [](uint8_t command, uint8_t ctrl_flag) {
    auto header_bytes = std::make_unique<uint8_t[]>(sizeof(RxGlobalHeader));
    auto *header = reinterpret_cast<RxGlobalHeader *>(&header_bytes[0]);
    header->msg_id = command;
    header->command = command;
    header->control_flags = ctrl_flag;
    return header_bytes;
  };

  auto parse = [](const std::vector<uint8_t> &v) {
    std::vector<CalibrationStatus> statuses;
    for (size_t i = 0; i < v.size(); i += EC_INPUT_FRAME_SIZE) {
      uint16_t d = (v.at(i + 1) << 8) | v.at(i);
      bool danger = (d & 0x1000) == 0x1000;
      uint16_t base = d & 0x0FFF;

      statuses.push_back(CalibrationStatus{danger, base});
    }
    return statuses;
  };

  auto succeed_calib = [](const std::vector<CalibrationStatus> &statuses) {
    uint16_t max_base = 0;
    for (auto status : statuses) {
      if (status.danger) return false;
      max_base = std::max(max_base, status.sync_base);
    }
    for (auto status : statuses) {
      if ((max_base - status.sync_base) % SYNC0_STEP != 0) return false;
    }
    return true;
  };

  std::vector<CalibrationStatus> statuses;
  size_t size;
  std::unique_ptr<uint8_t[]> body;
  std::vector<uint8_t> rx;

  // Change Sync0 interval to 1s
  auto calib_config = _config;
  calib_config.ec_sync0_cyctime_ns = MOD_PERIOD_MS * 1000 * 1000;
  _cnt->Close();
  _cnt->Open(_ifname.c_str(), _device_num, calib_config);

  // Negate is_sync_first_sync0 flag
  size = sizeof(RxGlobalHeader);
  body = make_calib_header(CMD_NEG_SYNC_FIRST_SYNC0, 0x00);
  Send(size, std::move(body));

  // Wait for synchronize modulation index in FPGA
  size_t t = static_cast<size_t>((_device_num - 1) / EC_DEVICE_PER_FRAME * EC_TRAFFIC_DELAY);
  std::this_thread::sleep_for(std::chrono::milliseconds((t + 2) * MOD_PERIOD_MS));
  while (_cnt->ec_dc_time() > 100 * 1000 * 1000) {  // Todo. 100ms is a magic number...
  }

  // Restore Sync0 interval
  _cnt->Close();
  _cnt->Open(_ifname.c_str(), _device_num, this->_config);

  // Read mod idx base
  size = sizeof(RxGlobalHeader);
  body = make_calib_header(CMD_READ_MOD_SYNC_BASE, 0x00);
  Send(size, std::move(body));

  // Wait for read mod idx base
  rx = WaitProcMsg(READ_MOD_IDX_BASE_HEADER, READ_MOD_IDX_BASE_HEADER_MASK);
  statuses = parse(rx);

  // Check status and get maximum mod idx base (for calculating shift)
  uint16_t max_base = 0;
  for (auto status : statuses) {
    if (status.danger) {
      std::cerr << "Calibration failed. " << std::endl;
      return false;
    }
    if (status.sync_base > MOD_BUF_SIZE) {
      std::cerr << "Calibration failed. Unexpected status has detected." << std::endl;
      return false;
    }
    max_base = std::max(max_base, status.sync_base);
  }

  // Shift mod idx base
  size = sizeof(RxGlobalHeader) + sizeof(uint16_t) * NUM_TRANS_IN_UNIT * _device_num;
  auto header = make_calib_header(CMD_SHIFT_MOD_SYNC_BASE, MOD_SYNC_BASE_SHIFT);
  body = std::make_unique<uint8_t[]>(size);
  std::memcpy(&body[0], &header[0], sizeof(RxGlobalHeader));
  auto *cursor = &body[0] + sizeof(RxGlobalHeader) / sizeof(body[0]);
  auto round_multiple = [](uint16_t number, uint16_t mul) { return ((number + mul / 2) / mul) * mul; };
  for (int i = 0; i < _device_num; i++) {
    int32_t diff = (max_base - statuses[i].sync_base) - round_multiple(max_base - statuses[i].sync_base, SYNC0_STEP);
    uint16_t shift = (MOD_BUF_SIZE + diff) % MOD_BUF_SIZE;
    uint8_t *src = reinterpret_cast<uint8_t *>(&shift);
    std::memcpy(cursor, src, sizeof(uint16_t));
    cursor += NUM_TRANS_IN_UNIT * sizeof(uint16_t) / sizeof(body[0]);
  }
  Send(size, std::move(body));

  // Wait for shift
  WaitProcMsg(CMD_SHIFT_MOD_SYNC_BASE, 0xFF);

  // Read mod idx base
  size = sizeof(RxGlobalHeader);
  body = make_calib_header(CMD_READ_MOD_SYNC_BASE, 0x00);
  Send(size, std::move(body));

  // Wait for read mod idx base
  rx = WaitProcMsg(READ_MOD_IDX_BASE_HEADER_AFTER_SHIFT, READ_MOD_IDX_BASE_HEADER_MASK);
  statuses = parse(rx);

  auto success = succeed_calib(statuses);
  if (!success) {
    std::cerr << "Failed to CalibrateModulation." << std::endl;
    std::cerr << "======= Modulation Log ========" << std::endl;
    for (size_t i = 0; i < statuses.size(); i++) {
      auto status = statuses[i];
      std::cerr << i << "," << static_cast<int>(status.danger) << "," << static_cast<int>(status.sync_base) << std::endl;
    }
    std::cerr << "===============================" << std::endl;
  }

  return success;
}
}  // namespace autd
