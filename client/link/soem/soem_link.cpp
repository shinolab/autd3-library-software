﻿// File: soem_link.cpp
// Project: soem
// Created Date: 08/03/2021
// Author: Shun Suzuki
// -----
// Last Modified: 20/05/2021
// Modified By: Shun Suzuki (suzuki@hapis.k.u-tokyo.ac.jp)
// -----
// Copyright (c) 2021 Hapis Lab. All rights reserved.
//

#include "soem_link.hpp"

#include "autdsoem.hpp"
#include "core/ec_config.hpp"

namespace autd::link {

std::vector<EtherCATAdapter> SOEMLink::enumerate_adapters() {
  std::vector<EtherCATAdapter> res;
  for (const auto& adapter : autdsoem::EtherCATAdapterInfo::enumerate_adapters()) res.emplace_back(adapter.desc, adapter.name);
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
  Error open() override;
  Error close() override;
  Error send(size_t size, const uint8_t* buf) override;
  Error read(uint8_t* rx, size_t buffer_len) override;
  bool is_open() override;

 private:
  autdsoem::SOEMController _cnt;
  size_t _device_num = 0;
  std::string _ifname;
  autdsoem::ECConfig _config{};
};

core::LinkPtr SOEMLink::create(const std::string& ifname, const size_t device_num) {
  core::LinkPtr link = std::make_shared<SOEMLinkImpl>(ifname, device_num);
  return link;
}

Error SOEMLinkImpl::open() {
  if (_ifname.empty()) return Err(std::string("Interface name is empty."));

  _config = autdsoem::ECConfig{};
  _config.ec_sm3_cycle_time_ns = core::EC_SM3_CYCLE_TIME_NANO_SEC;
  _config.ec_sync0_cycle_time_ns = core::EC_SYNC0_CYCLE_TIME_NANO_SEC;
  _config.header_size = core::HEADER_SIZE;
  _config.body_size = core::EC_OUTPUT_FRAME_SIZE - core::HEADER_SIZE;
  _config.input_frame_size = core::EC_INPUT_FRAME_SIZE;

  return _cnt.open(_ifname.c_str(), _device_num, _config);
}

Error SOEMLinkImpl::close() { return _cnt.close(); }

Error SOEMLinkImpl::send(const size_t size, const uint8_t* buf) { return _cnt.send(size, buf); }

Error SOEMLinkImpl::read(uint8_t* rx, [[maybe_unused]] size_t buffer_len) { return _cnt.read(rx); }

bool SOEMLinkImpl::is_open() { return _cnt.is_open(); }
}  // namespace autd::link
