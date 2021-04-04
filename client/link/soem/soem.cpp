﻿// File: soem_link.cpp
// Project: lib
// Created Date: 24/08/2019
// Author: Shun Suzuki
// -----
// Last Modified: 04/04/2021
// Modified By: Shun Suzuki (suzuki@hapis.k.u-tokyo.ac.jp)
// -----
// Copyright (c) 2019-2020 Hapis Lab. All rights reserved.
//

#include "link/soem.hpp"

#include <algorithm>
#include <bitset>
#include <chrono>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "../lib/ec_config.hpp"
#include "autdsoem.hpp"

namespace autd::link {

EtherCATAdapters SOEMLink::EnumerateAdapters(size_t* const size) {
  const auto adapters = autdsoem::EtherCATAdapterInfo::EnumerateAdapters();
  *size = adapters.size();
  EtherCATAdapters res;
  for (const auto& adapter : autdsoem::EtherCATAdapterInfo::EnumerateAdapters()) {
    EtherCATAdapter p;
    p.first = adapter.desc;
    p.second = adapter.name;
    res.emplace_back(p);
  }
  return res;
}

class SOEMLinkImpl final : public SOEMLink {
 public:
  SOEMLinkImpl(std::string ifname, const size_t device_num) : SOEMLink(), _device_num(device_num), _ifname(std::move(ifname)) {}
  ~SOEMLinkImpl() override = default;
  SOEMLinkImpl(const SOEMLinkImpl& v) noexcept = delete;
  SOEMLinkImpl& operator=(const SOEMLinkImpl& obj) = delete;
  SOEMLinkImpl(SOEMLinkImpl&& obj) = delete;
  SOEMLinkImpl& operator=(SOEMLinkImpl&& obj) = delete;

 protected:
  Result<bool, std::string> Open() override;
  Result<bool, std::string> Close() override;
  Result<bool, std::string> Send(size_t size, const uint8_t* buf) override;
  Result<bool, std::string> Read(uint8_t* rx, uint32_t buffer_len) override;
  bool is_open() override;

 private:
  autdsoem::SOEMController _cnt;
  size_t _device_num = 0;
  std::string _ifname;
  autdsoem::ECConfig _config{};
};

LinkPtr SOEMLink::Create(const std::string& ifname, const size_t device_num) {
  LinkPtr link = std::make_unique<SOEMLinkImpl>(ifname, device_num);
  return link;
}

Result<bool, std::string> SOEMLinkImpl::Open() {
  _config = autdsoem::ECConfig{};
  _config.ec_sm3_cycle_time_ns = EC_SM3_CYCLE_TIME_NANO_SEC;
  _config.ec_sync0_cycle_time_ns = EC_SYNC0_CYCLE_TIME_NANO_SEC;
  _config.header_size = HEADER_SIZE;
  _config.body_size = 498;
  _config.input_frame_size = EC_INPUT_FRAME_SIZE;

  return _cnt.Open(_ifname.c_str(), _device_num, _config);
}

Result<bool, std::string> SOEMLinkImpl::Close() { return _cnt.Close(); }

Result<bool, std::string> SOEMLinkImpl::Send(const size_t size, const uint8_t* buf) {
  if (!_cnt.is_open()) return Ok(false);

  return _cnt.Send(size, buf);
}

Result<bool, std::string> SOEMLinkImpl::Read(uint8_t* rx, [[maybe_unused]] uint32_t buffer_len) {
  if (!_cnt.is_open()) return Ok(false);

  return _cnt.Read(rx);
}

bool SOEMLinkImpl::is_open() { return _cnt.is_open(); }
}  // namespace autd::link
