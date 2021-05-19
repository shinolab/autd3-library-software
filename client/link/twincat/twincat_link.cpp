﻿// File: twincat_link.cpp
// Project: twincat
// Created Date: 08/03/2021
// Author: Shun Suzuki
// -----
// Last Modified: 19/05/2021
// Modified By: Shun Suzuki (suzuki@hapis.k.u-tokyo.ac.jp)
// -----
// Copyright (c) 2021 Hapis Lab. All rights reserved.
//

#include "twincat_link.hpp"

#include <sstream>
#include <string>

namespace autd::link {

struct AmsNetId {
  uint8_t b[6];
};

struct AmsAddr {
  AmsNetId net_id;
  uint16_t port;
};

class TwinCATLinkImpl final : public TwinCATLink {
 public:
  TwinCATLinkImpl() : _port(0) {}
  ~TwinCATLinkImpl() override = default;
  TwinCATLinkImpl(const TwinCATLinkImpl& v) noexcept = delete;
  TwinCATLinkImpl& operator=(const TwinCATLinkImpl& obj) = delete;
  TwinCATLinkImpl(TwinCATLinkImpl&& obj) = delete;
  TwinCATLinkImpl& operator=(TwinCATLinkImpl&& obj) = delete;

  Error open() override;
  Error close() override;
  Error send(size_t size, const uint8_t* buf) override;
  Error read(uint8_t* rx, size_t buffer_len) override;
  bool is_open() override;

 private:
  long _port;  // NOLINT
#ifdef _WIN32
  AmsNetId _net_id{};
  HMODULE _lib = nullptr;
#endif
};

core::LinkPtr TwinCATLink::create() { return std::make_shared<TwinCATLinkImpl>(); }

bool TwinCATLinkImpl::is_open() { return this->_port > 0; }

#ifdef _WIN32

constexpr uint32_t INDEX_GROUP = 0x3040030;
constexpr uint32_t INDEX_OFFSET_BASE = 0x81000000;
constexpr uint32_t INDEX_OFFSET_BASE_READ = 0x80000000;
constexpr uint16_t PORT = 301;

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

Error TwinCATLinkImpl::open() {
  this->_lib = LoadLibrary("TcAdsDll.dll");
  if (_lib == nullptr) return Err(std::string("couldn't find TcADS-DLL"));

  const auto port_open = reinterpret_cast<TcAdsPortOpenEx>(GetProcAddress(this->_lib, TCADS_ADS_PORT_OPEN_EX));
  this->_port = (*port_open)();
  if (!this->_port) return Err(std::string("Error: Failed to open a new ADS port"));

  AmsAddr addr{};
  const auto get_addr = reinterpret_cast<TcAdsGetLocalAddressEx>(GetProcAddress(this->_lib, TCADS_ADS_GET_LOCAL_ADDRESS_EX));
  if (const auto ret = get_addr(this->_port, &addr); ret) {
    std::stringstream ss;
    ss << "Error: AdsGetLocalAddress: " << std::hex << ret;
    return Err(ss.str());
  }

  this->_net_id = addr.net_id;
  return Ok();
}
Error TwinCATLinkImpl::close() {
  this->_port = 0;
  const auto port_close = reinterpret_cast<TcAdsPortCloseEx>(GetProcAddress(this->_lib, TCADS_ADS_PORT_CLOSE_EX));
  const auto res = (*port_close)(this->_port);
  if (res == 0) return Ok();
  std::stringstream ss;
  ss << "Error on closing (local): " << std::hex << res;
  return Err(ss.str());
}
Error TwinCATLinkImpl::send(const size_t size, const uint8_t* buf) {
  AmsAddr addr = {this->_net_id, PORT};
  const auto write = reinterpret_cast<TcAdsSyncWriteReqEx>(GetProcAddress(this->_lib, TCADS_ADS_SYNC_WRITE_REQ_EX));
  const auto ret = write(this->_port,  // NOLINT
                         &addr, INDEX_GROUP, INDEX_OFFSET_BASE,
                         static_cast<unsigned long>(size),  // NOLINT
                         const_cast<void*>(static_cast<const void*>(buf)));

  if (ret == 0) return Ok();
  // https://infosys.beckhoff.com/english.php?content=../content/1033/tcadscommon/html/tcadscommon_intro.htm&id=
  // 6 : target port not found
  std::stringstream ss;
  ss << "Error on sending data (local): " << std::hex << ret;
  return Err(ss.str());
}

Error TwinCATLinkImpl::read(uint8_t* rx, const size_t buffer_len) {
  AmsAddr addr = {this->_net_id, PORT};
  const auto read = reinterpret_cast<TcAdsSyncReadReqEx>(GetProcAddress(this->_lib, TCADS_ADS_SYNC_READ_REQ_EX));

  unsigned long read_bytes;           // NOLINT
  const auto ret = read(this->_port,  // NOLINT
                        &addr, INDEX_GROUP, INDEX_OFFSET_BASE_READ, static_cast<uint32_t>(buffer_len), rx, &read_bytes);
  if (ret == 0) return Ok();

  std::stringstream ss;
  ss << "Error on reading data: " << std::hex << ret;
  return Err(ss.str());
}

#else
Error TwinCATLinkImpl::open() {
  return Err(std::string("Link to localhost has not been compiled. Rebuild this library on a Twincat3 host machine with TcADS-DLL."));
}
Error TwinCATLinkImpl::close() { return Ok(); }
Error TwinCATLinkImpl::send(size_t size, const uint8_t* buf) {
  (void)size;
  (void)buf;
  return Ok();
}
Error TwinCATLinkImpl::read(uint8_t* rx, size_t buffer_len) {
  (void)rx;
  (void)buffer_len;
  return Ok();
}
#endif  // TC_ADS

}  // namespace autd::link
