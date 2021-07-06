// File: soem_link.cpp
// Project: soem
// Created Date: 08/03/2021
// Author: Shun Suzuki
// -----
// Last Modified: 04/07/2021
// Modified By: Shun Suzuki (suzuki@hapis.k.u-tokyo.ac.jp)
// -----
// Copyright (c) 2021 Hapis Lab. All rights reserved.
//

#include "autd3/core/ec_config.hpp"
#include "autd3/core/exception.hpp"
#include "autd3/link/soem.hpp"
#include "autdsoem.hpp"

namespace autd::link {

std::vector<EtherCATAdapter> SOEM::enumerate_adapters() {
  std::vector<EtherCATAdapter> res;
  for (const auto& adapter : autdsoem::EtherCATAdapterInfo::enumerate_adapters()) res.emplace_back(adapter.desc, adapter.name);
  return res;
}

class SOEMImpl final : public SOEM {
 public:
  SOEMImpl(std::string ifname, const size_t device_num, const uint32_t cycle_ticks)
      : SOEM(), _device_num(device_num), _cycle_ticks(cycle_ticks), _ifname(std::move(ifname)), _config() {}
  ~SOEMImpl() override = default;
  SOEMImpl(const SOEMImpl& v) noexcept = delete;
  SOEMImpl& operator=(const SOEMImpl& obj) = delete;
  SOEMImpl(SOEMImpl&& obj) = delete;
  SOEMImpl& operator=(SOEMImpl&& obj) = delete;

 protected:
  void open() override;
  void close() override;
  void send(const uint8_t* buf, size_t size) override;
  void read(uint8_t* rx, size_t buffer_len) override;
  bool is_open() override;

 private:
  autdsoem::SOEMController _cnt;
  size_t _device_num;
  uint32_t _cycle_ticks;
  std::string _ifname;
  autdsoem::ECConfig _config;
};

core::LinkPtr SOEM::create(const std::string& ifname, const size_t device_num, uint32_t cycle_ticks) {
  core::LinkPtr link = std::make_unique<SOEMImpl>(ifname, device_num, cycle_ticks);
  return link;
}

void SOEMImpl::open() {
  if (_ifname.empty()) throw core::LinkError("Interface name is empty");

  _config = autdsoem::ECConfig{};
  _config.ec_sm3_cycle_time_ns = core::EC_SM3_CYCLE_TIME_NANO_SEC * _cycle_ticks;
  _config.ec_sync0_cycle_time_ns = core::EC_SYNC0_CYCLE_TIME_NANO_SEC * _cycle_ticks;
  _config.header_size = core::HEADER_SIZE;
  _config.body_size = core::EC_OUTPUT_FRAME_SIZE - core::HEADER_SIZE;
  _config.input_frame_size = core::EC_INPUT_FRAME_SIZE;

  return _cnt.open(_ifname.c_str(), _device_num, _config);
}

void SOEMImpl::close() { return _cnt.close(); }

void SOEMImpl::send(const uint8_t* buf, const size_t size) { return _cnt.send(buf, size); }

void SOEMImpl::read(uint8_t* rx, [[maybe_unused]] size_t buffer_len) { return _cnt.read(rx); }

bool SOEMImpl::is_open() { return _cnt.is_open(); }
}  // namespace autd::link
