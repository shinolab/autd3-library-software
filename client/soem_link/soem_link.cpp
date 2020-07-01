// File: soem_link.cpp
// Project: lib
// Created Date: 24/08/2019
// Author: Shun Suzuki
// -----
// Last Modified: 01/07/2020
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

#include "../lib/ec_config.hpp"
#include "../lib/privdef.hpp"
#include "autdsoem.hpp"

namespace autd::link {

EtherCATAdapters SOEMLink::EnumerateAdapters(int *const size) {
  auto adapters = autdsoem::EtherCATAdapterInfo::EnumerateAdapters();
  *size = static_cast<int>(adapters.size());
  EtherCATAdapters res;
  for (auto adapter : autdsoem::EtherCATAdapterInfo::EnumerateAdapters()) {
    EtherCATAdapter p;
    p.first = adapter.desc;
    p.second = adapter.name;
    res.push_back(p);
  }
  return res;
}

class SOEMLinkImpl : public SOEMLink {
 public:
  ~SOEMLinkImpl() override {}

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
};

LinkPtr SOEMLink::Create(std::string ifname, int device_num) {
  auto link = std::make_shared<SOEMLinkImpl>();
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
}  // namespace autd::link
