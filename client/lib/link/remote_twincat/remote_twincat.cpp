// File: twincat_link.cpp
// Project: twincat
// Created Date: 08/03/2021
// Author: Shun Suzuki
// -----
// Last Modified: 13/09/2021
// Modified By: Shun Suzuki (suzuki@hapis.k.u-tokyo.ac.jp)
// -----
// Copyright (c) 2021 Hapis Lab. All rights reserved.
//

#include <AdsLib.h>

#if _WINDOWS
#include <Windows.h>
#endif

#include <sstream>
#include <string>
#include <vector>

#include "autd3/core/exception.hpp"
#include "autd3/core/link.hpp"
#include "autd3/link/remote_twincat.hpp"

namespace autd::link {

namespace {

std::vector<std::string> split(const std::string& s, const char deliminator) {
  std::vector<std::string> tokens;
  std::string token;
  for (const auto& ch : s) {
    if (ch == deliminator) {
      if (!token.empty()) tokens.emplace_back(token);
      token.clear();
    } else {
      token += ch;
    }
  }
  if (!token.empty()) tokens.emplace_back(token);
  return tokens;
}
}  // namespace

constexpr uint32_t INDEX_GROUP = 0x3040030;
constexpr uint32_t INDEX_OFFSET_BASE = 0x81000000;
constexpr uint32_t INDEX_OFFSET_BASE_READ = 0x80000000;
constexpr uint16_t PORT = 301;

class RemoteTwinCATImpl final : public RemoteTwinCAT {
 public:
  RemoteTwinCATImpl(std::string ipv4_addr, std::string remote_ams_net_id, std::string local_ams_net_id)
      : RemoteTwinCAT(),
        _local_ams_net_id(std::move(local_ams_net_id)),
        _remote_ams_net_id(std::move(remote_ams_net_id)),
        _ipv4_addr(std::move(ipv4_addr)) {}
  ~RemoteTwinCATImpl() override = default;
  RemoteTwinCATImpl(const RemoteTwinCATImpl& v) noexcept = delete;
  RemoteTwinCATImpl& operator=(const RemoteTwinCATImpl& obj) = delete;
  RemoteTwinCATImpl(RemoteTwinCATImpl&& obj) = delete;
  RemoteTwinCATImpl& operator=(RemoteTwinCATImpl&& obj) = delete;

  void open() override;
  void close() override;
  void send(const uint8_t* buf, size_t size) override;
  void read(uint8_t* rx, size_t buffer_len) override;
  bool is_open() override;

 private:
  std::string _local_ams_net_id;
  std::string _remote_ams_net_id;
  std::string _ipv4_addr;
  long _port = 0L;  // NOLINT
  AmsNetId _net_id;
};

core::LinkPtr RemoteTwinCAT::create(const std::string& ipv4_addr, const std::string& remote_ams_net_id, const std::string& local_ams_net_id) {
  return std::make_unique<RemoteTwinCATImpl>(ipv4_addr, remote_ams_net_id, local_ams_net_id);
}

void RemoteTwinCATImpl::open() {
  const auto octets = split(_remote_ams_net_id, '.');
  if (octets.size() != 6) throw core::exception::LinkError("Ams net id must have 6 octets");

  if (_ipv4_addr.empty()) {
    for (auto i = 0; i < 3; i++) _ipv4_addr += octets[i] + ".";
    _ipv4_addr += octets[3];
  }

  if (!_local_ams_net_id.empty()) {
    const auto local_octets = split(_local_ams_net_id, '.');
    if (local_octets.size() != 6) throw core::exception::LinkError("Ams net id must have 6 octets");
    bhf::ads::SetLocalAddress({static_cast<uint8_t>(std::stoi(local_octets[0])), static_cast<uint8_t>(std::stoi(local_octets[1])),
                               static_cast<uint8_t>(std::stoi(local_octets[2])), static_cast<uint8_t>(std::stoi(local_octets[3])),
                               static_cast<uint8_t>(std::stoi(local_octets[4])), static_cast<uint8_t>(std::stoi(local_octets[5]))});
  }

  this->_net_id = {static_cast<uint8_t>(std::stoi(octets[0])), static_cast<uint8_t>(std::stoi(octets[1])),
                   static_cast<uint8_t>(std::stoi(octets[2])), static_cast<uint8_t>(std::stoi(octets[3])),
                   static_cast<uint8_t>(std::stoi(octets[4])), static_cast<uint8_t>(std::stoi(octets[5]))};

  if (AdsAddRoute(this->_net_id, _ipv4_addr.c_str()) != 0) throw core::exception::LinkError("Could not connect to remote");

  this->_port = AdsPortOpenEx();

  if (this->_port == 0) throw core::exception::LinkError("Failed to open a new ADS port");
}

void RemoteTwinCATImpl::close() {
  if (AdsPortCloseEx(this->_port) != 0) throw core::exception::LinkError("Failed to close");
  this->_port = 0;
}

void RemoteTwinCATImpl::send(const uint8_t* buf, const size_t size) {
  const AmsAddr p_addr = {this->_net_id, PORT};
  const auto ret = AdsSyncWriteReqEx(this->_port, &p_addr, INDEX_GROUP, INDEX_OFFSET_BASE, static_cast<uint32_t>(size), buf);
  if (ret == 0) return;

  std::stringstream ss;
  if (ret == ADSERR_DEVICE_INVALIDSIZE)
    ss << "The number of devices is invalid.";
  else
    ss << "Error on sending data: " << std::hex << ret;
  throw core::exception::LinkError(ss.str());
}

void RemoteTwinCATImpl::read(uint8_t* rx, const size_t buffer_len) {
  const AmsAddr p_addr = {this->_net_id, PORT};
  uint32_t read_bytes;
  const auto ret = AdsSyncReadReqEx2(this->_port, &p_addr, INDEX_GROUP, INDEX_OFFSET_BASE_READ, static_cast<uint32_t>(buffer_len), rx, &read_bytes);
  if (ret == 0) return;

  std::stringstream ss;
  ss << "Error on reading data: " << std::hex << ret;
  throw core::exception::LinkError(ss.str());
}

bool RemoteTwinCATImpl::is_open() { return this->_port > 0; }

}  // namespace autd::link
