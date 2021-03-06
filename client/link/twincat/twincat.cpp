﻿// File: ethercat_link.cpp
// Project: lib
// Created Date: 01/06/2016
// Author: Seki Inoue
// -----
// Last Modified: 26/12/2020
// Modified By: Shun Suzuki (suzuki@hapis.k.u-tokyo.ac.jp)
// -----
// Copyright (c) 2016-2020 Hapis Lab. All rights reserved.
//

#include "link/twincat.hpp"

#if _WINDOWS
#include <codeanalysis/warnings.h>  // NOLINT
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

namespace autd::link {

constexpr uint32_t INDEX_GROUP = 0x3040030;
constexpr uint32_t INDEX_OFFSET_BASE = 0x81000000;
constexpr uint32_t INDEX_OFFSET_BASE_READ = 0x80000000;
constexpr uint16_t PORT = 301;

static std::vector<std::string> Split(const std::string& s, const char deliminator) {
  std::vector<std::string> tokens;
  std::string token;
  for (auto ch : s) {
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

class TwinCATLinkImpl final : public TwinCATLink {
 public:
  TwinCATLinkImpl(std::string ipv4_addr, std::string ams_net_id)
      : TwinCATLink(), _ams_net_id(std::move(ams_net_id)), _ipv4_addr(std::move(ipv4_addr)) {}
  ~TwinCATLinkImpl() override = default;
  TwinCATLinkImpl(const TwinCATLinkImpl& v) noexcept = delete;
  TwinCATLinkImpl& operator=(const TwinCATLinkImpl& obj) = delete;
  TwinCATLinkImpl(TwinCATLinkImpl&& obj) = delete;
  TwinCATLinkImpl& operator=(TwinCATLinkImpl&& obj) = delete;

  void Open() override;
  void Close() override;
  std::optional<int32_t> Send(size_t size, std::unique_ptr<uint8_t[]> buf) override;
  std::optional<int32_t> Read(uint8_t* rx, uint32_t buffer_len) override;
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
  auto octets = Split(_ams_net_id, '.');
  if (octets.size() != 6) {
    std::cerr << "Ams net id must have 6 octets\n";
    return;
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

bool TwinCATLinkImpl::is_open() { return this->_port > 0; }

std::optional<int32_t> TwinCATLinkImpl::Send(const size_t size, const std::unique_ptr<uint8_t[]> buf) {
  const AmsAddr p_addr = {this->_net_id, PORT};
  const auto ret = AdsSyncWriteReqEx(this->_port,  // NOLINT
                                     &p_addr, INDEX_GROUP, INDEX_OFFSET_BASE, static_cast<uint32_t>(size), &buf[0]);
  if (ret > 0) {
    if (ret == ADSERR_DEVICE_INVALIDSIZE) {
      std::cerr << "The number of devices is invalid." << std::endl;

    } else {
      std::cerr << "Error on sending data: " << std::hex << ret << std::endl;
    }
    return static_cast<int32_t>(ret);
  }

  return std::nullopt;
}

std::optional<int32_t> TwinCATLinkImpl::Read(uint8_t* rx, const uint32_t buffer_len) {
  const AmsAddr p_addr = {this->_net_id, PORT};
  const auto buffer = std::make_unique<uint8_t[]>(buffer_len);
  uint32_t read_bytes;
  const auto ret = AdsSyncReadReqEx2(this->_port,  // NOLINT
                                     &p_addr, INDEX_GROUP, INDEX_OFFSET_BASE_READ, buffer_len, rx, &read_bytes);

  if (ret > 0) {
    std::cerr << "Error on reading data: " << std::hex << ret << std::endl;
    return static_cast<int32_t>(ret);
  }

  return std::nullopt;
}

class LocalTwinCATLinkImpl final : public LocalTwinCATLink {
 public:
  LocalTwinCATLinkImpl() : LocalTwinCATLink() {}
  ~LocalTwinCATLinkImpl() override = default;
  LocalTwinCATLinkImpl(const LocalTwinCATLinkImpl& v) noexcept = delete;
  LocalTwinCATLinkImpl& operator=(const LocalTwinCATLinkImpl& obj) = delete;
  LocalTwinCATLinkImpl(LocalTwinCATLinkImpl&& obj) = delete;
  LocalTwinCATLinkImpl& operator=(LocalTwinCATLinkImpl&& obj) = delete;

 protected:
  void Open() override;
  void Close() override;
  std::optional<int32_t> Send(size_t size, std::unique_ptr<uint8_t[]> buf) override;
  std::optional<int32_t> Read(uint8_t* rx, uint32_t buffer_len) override;
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

bool LocalTwinCATLinkImpl::is_open() { return this->_port > 0; }

#ifdef _WIN32
typedef long(_stdcall* TcAdsPortOpenEx)();                       // NOLINT
typedef long(_stdcall* TcAdsPortCloseEx)(long);                  // NOLINT
typedef long(_stdcall* TcAdsGetLocalAddressEx)(long, AmsAddr*);  // NOLINT
typedef long(_stdcall* TcAdsSyncWriteReqEx)(long, AmsAddr*,      // NOLINT
                                            unsigned long,       // NOLINT
                                            unsigned long,       // NOLINT
                                            unsigned long,       // NOLINT
                                            void*);              // NOLINT
typedef long(_stdcall* TcAdsSyncReadReqEx)(long, AmsAddr*,       // NOLINT
                                           unsigned long,        // NOLINT
                                           unsigned long,        // NOLINT
                                           unsigned long,        // NOLINT
                                           void*,                // NOLINT
                                           unsigned long*);      // NOLINT

constexpr auto TCADS_ADS_PORT_OPEN_EX = "AdsPortOpenEx";
constexpr auto TCADS_ADS_GET_LOCAL_ADDRESS_EX = "AdsGetLocalAddressEx";
constexpr auto TCADS_ADS_PORT_CLOSE_EX = "AdsPortCloseEx";
constexpr auto TCADS_ADS_SYNC_WRITE_REQ_EX = "AdsSyncWriteReqEx";
constexpr auto TCADS_ADS_SYNC_READ_REQ_EX = "AdsSyncReadReqEx2";

void LocalTwinCATLinkImpl::Open() {
  this->_lib = LoadLibrary("TcAdsDll.dll");
  if (_lib == nullptr) {
    std::cerr << "couldn't find TcADS-DLL.\n";
    return;
  }
  // open a new ADS port
  const auto port_open = reinterpret_cast<TcAdsPortOpenEx>(GetProcAddress(this->_lib, TCADS_ADS_PORT_OPEN_EX));
  this->_port = (*port_open)();
  if (!this->_port) {
    std::cerr << "Error: Failed to open a new ADS port." << std::endl;
  }
  AmsAddr addr;
  const auto get_addr = reinterpret_cast<TcAdsGetLocalAddressEx>(GetProcAddress(this->_lib, TCADS_ADS_GET_LOCAL_ADDRESS_EX));
  const auto n_err = get_addr(this->_port, &addr);  // NOLINT
  if (n_err) std::cerr << "Error: AdsGetLocalAddress: " << n_err << std::endl;
  this->_net_id = addr.netId;
}
void LocalTwinCATLinkImpl::Close() {
  this->_port = 0;
  const auto port_close = reinterpret_cast<TcAdsPortCloseEx>(GetProcAddress(this->_lib, TCADS_ADS_PORT_CLOSE_EX));
  (*port_close)(this->_port);
}
std::optional<int32_t> LocalTwinCATLinkImpl::Send(const size_t size, const std::unique_ptr<uint8_t[]> buf) {
  AmsAddr addr = {this->_net_id, PORT};
  const auto write = reinterpret_cast<TcAdsSyncWriteReqEx>(GetProcAddress(this->_lib, TCADS_ADS_SYNC_WRITE_REQ_EX));
  const auto ret = write(this->_port,  // NOLINT
                         &addr, INDEX_GROUP, INDEX_OFFSET_BASE,
                         static_cast<unsigned long>(size),  // NOLINT
                         &buf[0]);
  if (ret > 0) {
    // https://infosys.beckhoff.com/english.php?content=../content/1033/tcadscommon/html/tcadscommon_intro.htm&id=
    // 6 : target port not found
    std::cerr << "Error on sending data (local): " << std::hex << ret << std::endl;
    return static_cast<int32_t>(ret);
  }

  return std::nullopt;
}

std::optional<int32_t> LocalTwinCATLinkImpl::Read(uint8_t* rx, const uint32_t buffer_len) {
  AmsAddr addr = {this->_net_id, PORT};
  const auto read = reinterpret_cast<TcAdsSyncReadReqEx>(GetProcAddress(this->_lib, TCADS_ADS_SYNC_READ_REQ_EX));

  unsigned long read_bytes;           // NOLINT
  const auto ret = read(this->_port,  // NOLINT
                        &addr, INDEX_GROUP, INDEX_OFFSET_BASE_READ, buffer_len, rx, &read_bytes);

  if (ret > 0) {
    std::cerr << "Error on reading data: " << std::hex << ret << std::endl;
    return static_cast<int>(ret);
  }

  return std::nullopt;
}

#else
void LocalTwinCATLinkImpl::Open() {
  std::cerr << "Link to localhost has not been compiled. Rebuild this library on a Twincat3 host machine with TcADS-DLL.\n";
  return;
}
void LocalTwinCATLinkImpl::Close() {}
std::optional<int32_t> LocalTwinCATLinkImpl::Send(size_t size, std::unique_ptr<uint8_t[]> buf) { return std::nullopt; }
std::optional<int32_t> LocalTwinCATLinkImpl::Read(uint8_t* rx, uint32_t buffer_len) { return std::nullopt; }
#endif  // TC_ADS

}  // namespace autd::link
