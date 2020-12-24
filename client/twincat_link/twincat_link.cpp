// File: ethercat_link.cpp
// Project: lib
// Created Date: 01/06/2016
// Author: Seki Inoue
// -----
// Last Modified: 24/12/2020
// Modified By: Shun Suzuki (suzuki@hapis.k.u-tokyo.ac.jp)
// -----
// Copyright (c) 2016-2020 Hapis Lab. All rights reserved.
//

#include "twincat_link.hpp"

#if _WINDOWS
#include <codeanalysis/warnings.h>
#pragma warning(push)
#pragma warning(disable : ALL_CODE_ANALYSIS_WARNINGS)
#endif
#include <AdsLib.h>
#if _WINDOWS
#pragma warning(pop)
#define NOMINMAX
#include <Windows.h>
#include <winnt.h>
#else
typedef void* HMODULE;
#endif

#include <exception>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "../lib/privdef.hpp"

namespace autd::link {

constexpr uint32_t INDEX_GROUP = 0x3040030;
constexpr uint32_t INDEX_OFFSET_BASE = 0x81000000;
constexpr uint32_t INDEX_OFFSET_BASE_READ = 0x80000000;
constexpr uint16_t PORT = 301;

static std::vector<std::string> split(const std::string& s, const char deliminator) {
  std::vector<std::string> tokens;
  std::string token;
  for (auto ch : s) {
    if (ch == deliminator) {
      if (!token.empty()) tokens.push_back(token);
      token.clear();
    } else {
      token += ch;
    }
  }
  if (!token.empty()) tokens.push_back(token);
  return tokens;
}

class TwinCATLinkImpl final : public TwinCATLink {
 public:
  TwinCATLinkImpl(std::string ipv4_addr, std::string ams_net_id)
      : TwinCATLink(), _ams_net_id(std::move(ams_net_id)), _ipv4_addr(std::move(ipv4_addr)) {}
  ~TwinCATLinkImpl() override = default;
  TwinCATLinkImpl(const TwinCATLinkImpl& v) noexcept = default;
  TwinCATLinkImpl& operator=(const TwinCATLinkImpl& obj) = default;
  TwinCATLinkImpl(TwinCATLinkImpl&& obj) = default;
  TwinCATLinkImpl& operator=(TwinCATLinkImpl&& obj) = default;

  void Open() override;
  void Close() override;
  void Send(size_t size, std::unique_ptr<uint8_t[]> buf) override;
  std::vector<uint8_t> Read(uint32_t buffer_len) override;
  bool is_open() override;

 private:
  std::string _ams_net_id;
  std::string _ipv4_addr;
  long _port = 0L;  // NOLINT
  AmsNetId _net_id;
};

LinkPtr TwinCATLink::Create(const std::string& ams_net_id) { return Create("", ams_net_id); }

LinkPtr TwinCATLink::Create(const std::string& ipv4_addr, const std::string& ams_net_id) {
  LinkPtr link = std::make_unique<TwinCATLinkImpl>(ipv4_addr, ams_net_id);
  return link;
}

void TwinCATLinkImpl::Open() {
  auto octets = split(_ams_net_id, '.');
  if (octets.size() != 6) {
    throw std::runtime_error("Ams net id must have 6 octets");
  }
  if (_ipv4_addr.empty()) {
    // Extract ipv6 addr from leading four octets of the ams net id.
    for (auto i = 0; i < 3; i++) {
      _ipv4_addr += octets[i] + ".";
    }
    _ipv4_addr += octets[3];
  }
  this->_net_id = {static_cast<uint8_t>(std::stoi(octets[0])), static_cast<uint8_t>(std::stoi(octets[1])),
                   static_cast<uint8_t>(std::stoi(octets[2])), static_cast<uint8_t>(std::stoi(octets[3])),
                   static_cast<uint8_t>(std::stoi(octets[4])), static_cast<uint8_t>(std::stoi(octets[5]))};
  if (AdsAddRoute(this->_net_id, _ipv4_addr.c_str())) {
    std::cerr << "Error: Could not connect to remote." << std::endl;
    return;
  }
  // open a new ADS port
  this->_port = AdsPortOpenEx();
  if (!this->_port) {
    std::cerr << "Error: Failed to open a new ADS port." << std::endl;
  }
}

void TwinCATLinkImpl::Close() {
  this->_port = 0;
  AdsPortCloseEx(this->_port);
}

bool TwinCATLinkImpl::is_open() { return (this->_port > 0); }

void TwinCATLinkImpl::Send(const size_t size, const std::unique_ptr<uint8_t[]> buf) {
  const AmsAddr p_addr = {this->_net_id, PORT};
  const auto ret = AdsSyncWriteReqEx(this->_port,  // NOLINT
                                     &p_addr, INDEX_GROUP, INDEX_OFFSET_BASE, static_cast<uint32_t>(size), &buf[0]);
  if (ret > 0) {
    if (ret == ADSERR_DEVICE_INVALIDSIZE) {
      std::cerr << "The number of devices is invalid." << std::endl;

    } else {
      std::cerr << "Error on sending data: " << std::hex << ret << std::endl;
    }
    throw static_cast<int>(ret);
  }
}

std::vector<uint8_t> TwinCATLinkImpl::Read(const uint32_t buffer_len) {
  const AmsAddr p_addr = {this->_net_id, PORT};
  const auto buffer = std::make_unique<uint8_t[]>(buffer_len);
  uint32_t read_bytes;
  const auto ret = AdsSyncReadReqEx2(this->_port,  // NOLINT
                                     &p_addr, INDEX_GROUP, INDEX_OFFSET_BASE_READ, buffer_len, &buffer[0], &read_bytes);

  if (ret > 0) {
    std::cerr << "Error on reading data: " << std::hex << ret << std::endl;
    throw static_cast<int>(ret);
  }

  std::vector<uint8_t> res(&buffer[0], &buffer[read_bytes]);
  return res;
}

class LocalTwinCATLinkImpl final : public LocalTwinCATLink {
 public:
  LocalTwinCATLinkImpl() : LocalTwinCATLink() {}
  ~LocalTwinCATLinkImpl() override = default;
  LocalTwinCATLinkImpl(const LocalTwinCATLinkImpl& v) noexcept = default;
  LocalTwinCATLinkImpl& operator=(const LocalTwinCATLinkImpl& obj) = default;
  LocalTwinCATLinkImpl(LocalTwinCATLinkImpl&& obj) = default;
  LocalTwinCATLinkImpl& operator=(LocalTwinCATLinkImpl&& obj) = default;

 protected:
  void Open() override;
  void Close() override;
  void Send(size_t size, std::unique_ptr<uint8_t[]> buf) override;
  std::vector<uint8_t> Read(uint32_t buffer_len) override;
  bool is_open() override;

 private:
  std::string _ams_net_id;
  std::string _ipv4_addr;
  long _port = 0L;  // NOLINT
  AmsNetId _net_id;
  HMODULE _lib = nullptr;
};

LinkPtr LocalTwinCATLink::Create() {
  LinkPtr link = std::make_unique<LocalTwinCATLinkImpl>();
  return link;
}

bool LocalTwinCATLinkImpl::is_open() { return (this->_port > 0); }

#ifdef _WIN32
typedef long(_stdcall* tc_ads_port_open_ex)();                        // NOLINT
typedef long(_stdcall* tc_ads_port_close_ex)(long);                   // NOLINT
typedef long(_stdcall* tc_ads_get_local_address_ex)(long, AmsAddr*);  // NOLINT
typedef long(_stdcall* tc_ads_sync_write_req_ex)(long, AmsAddr*,      // NOLINT
                                                 unsigned long,       // NOLINT
                                                 unsigned long,       // NOLINT
                                                 unsigned long,       // NOLINT
                                                 void*);              // NOLINT
typedef long(_stdcall* tc_ads_sync_read_req_ex)(long, AmsAddr*,       // NOLINT
                                                unsigned long,        // NOLINT
                                                unsigned long,        // NOLINT
                                                unsigned long,        // NOLINT
                                                void*,                // NOLINT
                                                unsigned long*);      // NOLINT
#ifdef _WIN64
constexpr auto TCADS_ADS_PORT_OPEN_EX = "AdsPortOpenEx";
constexpr auto TCADS_ADS_GET_LOCAL_ADDRESS_EX = "AdsGetLocalAddressEx";
constexpr auto TCADS_ADS_PORT_CLOSE_EX = "AdsPortCloseEx";
constexpr auto TCADS_ADS_SYNC_WRITE_REQ_EX = "AdsSyncWriteReqEx";
constexpr auto TCADS_ADS_SYNC_READ_REQ_EX = "AdsSyncReadReqEx2";
#endif

void LocalTwinCATLinkImpl::Open() {
  this->_lib = LoadLibrary("TcAdsDll.dll");
  if (_lib == nullptr) {
    throw std::runtime_error("couldn't find TcADS-DLL.");
  }
  // open a new ADS port
  const auto port_open = reinterpret_cast<tc_ads_port_open_ex>(GetProcAddress(this->_lib, TCADS_ADS_PORT_OPEN_EX));
  this->_port = (*port_open)();
  if (!this->_port) {
    std::cerr << "Error: Failed to open a new ADS port." << std::endl;
  }
  AmsAddr addr;
  const auto getAddr = reinterpret_cast<tc_ads_get_local_address_ex>(GetProcAddress(this->_lib, TCADS_ADS_GET_LOCAL_ADDRESS_EX));
  const auto n_err = getAddr(this->_port, &addr);  // NOLINT
  if (n_err) std::cerr << "Error: AdsGetLocalAddress: " << n_err << std::endl;
  this->_net_id = addr.netId;
}
void LocalTwinCATLinkImpl::Close() {
  this->_port = 0;
  const auto port_close = reinterpret_cast<tc_ads_port_close_ex>(GetProcAddress(this->_lib, TCADS_ADS_PORT_CLOSE_EX));
  (*port_close)(this->_port);
}
void LocalTwinCATLinkImpl::Send(const size_t size, const std::unique_ptr<uint8_t[]> buf) {
  AmsAddr addr = {this->_net_id, PORT};
  const auto write = reinterpret_cast<tc_ads_sync_write_req_ex>(GetProcAddress(this->_lib, TCADS_ADS_SYNC_WRITE_REQ_EX));
  const auto ret = write(this->_port,  // NOLINT
                         &addr, INDEX_GROUP, INDEX_OFFSET_BASE,
                         static_cast<unsigned long>(size),  // NOLINT
                         &buf[0]);
  if (ret > 0) {
    // https://infosys.beckhoff.com/english.php?content=../content/1033/tcadscommon/html/tcadscommon_intro.htm&id=
    // 6 : target port not found
    std::cerr << "Error on sending data (local): " << std::hex << ret << std::endl;
    throw static_cast<int>(ret);
  }
}

std::vector<uint8_t> LocalTwinCATLinkImpl::Read(uint32_t buffer_len) {
  AmsAddr addr = {this->_net_id, PORT};
  const auto read = reinterpret_cast<tc_ads_sync_read_req_ex>(GetProcAddress(this->_lib, TCADS_ADS_SYNC_READ_REQ_EX));

  const auto buffer = std::make_unique<uint8_t[]>(buffer_len);
  unsigned long read_bytes;           // NOLINT
  const auto ret = read(this->_port,  // NOLINT
                        &addr, INDEX_GROUP, INDEX_OFFSET_BASE_READ, buffer_len, &buffer[0], &read_bytes);

  if (ret > 0) {
    std::cerr << "Error on reading data: " << std::hex << ret << std::endl;
    throw static_cast<int>(ret);
  }

  std::vector<uint8_t> res(&buffer[0], &buffer[read_bytes]);
  return res;
}

#else
void LocalTwinCATLinkImpl::Open() {
  throw std::runtime_error(
      "Link to localhost has not been compiled. Rebuild this library on a "
      "Twincat3 host machine with TcADS-DLL.");
}
void LocalTwinCATLinkImpl::Close() {}
void LocalTwinCATLinkImpl::Send(size_t size, std::unique_ptr<uint8_t[]> buf) {}
std::vector<uint8_t> LocalTwinCATLinkImpl::Read(uint32_t buffer_len) { return std::vector<uint8_t>(); }
#endif  // TC_ADS

}  // namespace autd::link
