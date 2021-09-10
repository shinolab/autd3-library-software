// File: twincat_link.cpp
// Project: twincat
// Created Date: 08/03/2021
// Author: Shun Suzuki
// -----
// Last Modified: 10/09/2021
// Modified By: Shun Suzuki (suzuki@hapis.k.u-tokyo.ac.jp)
// -----
// Copyright (c) 2021 Hapis Lab. All rights reserved.
//

#if _WINDOWS
#include <Windows.h>
#endif

#include <sstream>
#include <string>

#include "autd3/core/exception.hpp"
#include "autd3/core/link.hpp"
#include "autd3/link/twincat.hpp"

namespace autd::link {

struct AmsNetId {
  uint8_t b[6];
};

struct AmsAddr {
  AmsNetId net_id;
  uint16_t port;
};

class TwinCATImpl final : public TwinCAT {
 public:
  TwinCATImpl() : _port(0) {}
  ~TwinCATImpl() override = default;
  TwinCATImpl(const TwinCATImpl& v) noexcept = delete;
  TwinCATImpl& operator=(const TwinCATImpl& obj) = delete;
  TwinCATImpl(TwinCATImpl&& obj) = delete;
  TwinCATImpl& operator=(TwinCATImpl&& obj) = delete;

  void open() override;
  void close() override;
  void send(const uint8_t* buf, size_t size) override;
  void read(uint8_t* rx, size_t buffer_len) override;
  bool is_open() override;

 private:
  long _port;  // NOLINT
#ifdef _WIN32
  AmsNetId _net_id{};
  HMODULE _lib = nullptr;
#endif
};

core::LinkPtr TwinCAT::create() { return std::make_unique<TwinCATImpl>(); }

bool TwinCATImpl::is_open() { return this->_port > 0; }

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

void TwinCATImpl::open() {
  this->_lib = LoadLibrary("TcAdsDll.dll");
  if (_lib == nullptr) throw core::LinkError("couldn't find TcADS-DLL");

  const auto port_open = reinterpret_cast<TcAdsPortOpenEx>(GetProcAddress(this->_lib, TCADS_ADS_PORT_OPEN_EX));
  this->_port = (*port_open)();
  if (!this->_port) throw core::LinkError("failed to open a new ADS port");

  AmsAddr addr{};
  const auto get_addr = reinterpret_cast<TcAdsGetLocalAddressEx>(GetProcAddress(this->_lib, TCADS_ADS_GET_LOCAL_ADDRESS_EX));
  if (const auto ret = get_addr(this->_port, &addr); ret) {
    std::stringstream ss;
    ss << "AdsGetLocalAddress: " << std::hex << ret;
    throw core::LinkError(ss.str());
  }

  this->_net_id = addr.net_id;
}

void TwinCATImpl::close() {
  if (!this->is_open()) return;

  const auto port_close = reinterpret_cast<TcAdsPortCloseEx>(GetProcAddress(this->_lib, TCADS_ADS_PORT_CLOSE_EX));
  const auto res = (*port_close)(this->_port);
  if (res == 0) {
    this->_port = 0;
    return;
  }
  std::stringstream ss;
  ss << "Error on closing (local): " << std::hex << res;
  throw core::LinkError(ss.str());
}

void TwinCATImpl::send(const uint8_t* buf, const size_t size) {
  if (!this->is_open()) throw core::LinkError("Link is closed");

  AmsAddr addr = {this->_net_id, PORT};
  const auto write = reinterpret_cast<TcAdsSyncWriteReqEx>(GetProcAddress(this->_lib, TCADS_ADS_SYNC_WRITE_REQ_EX));
  const auto ret = write(this->_port,  // NOLINT
                         &addr, INDEX_GROUP, INDEX_OFFSET_BASE,
                         static_cast<unsigned long>(size),  // NOLINT
                         const_cast<void*>(static_cast<const void*>(buf)));

  if (ret == 0) return;
  // https://infosys.beckhoff.com/english.php?content=../content/1033/tcadscommon/html/tcadscommon_intro.htm&id=
  // 6 : target port not found
  std::stringstream ss;
  ss << "Error on sending data (local): " << std::hex << ret;
  throw core::LinkError(ss.str());
}

void TwinCATImpl::read(uint8_t* rx, const size_t buffer_len) {
  if (!this->is_open()) throw core::LinkError("Link is closed");

  AmsAddr addr = {this->_net_id, PORT};
  const auto read = reinterpret_cast<TcAdsSyncReadReqEx>(GetProcAddress(this->_lib, TCADS_ADS_SYNC_READ_REQ_EX));

  unsigned long read_bytes;           // NOLINT
  const auto ret = read(this->_port,  // NOLINT
                        &addr, INDEX_GROUP, INDEX_OFFSET_BASE_READ, static_cast<uint32_t>(buffer_len), rx, &read_bytes);
  if (ret == 0) return;

  std::stringstream ss;
  ss << "Error on reading data: " << std::hex << ret;
  throw core::LinkError(ss.str());
}

#else
void TwinCATImpl::open() {
  throw core::LinkError("Link to localhost has not been compiled. Rebuild this library on a Twincat3 host machine with TcADS-DLL.");
}
void TwinCATImpl::close() { return; }
void TwinCATImpl::send(const uint8_t*, size_t) { return; }
void TwinCATImpl::read(uint8_t*, size_t) { return; }
#endif  // TC_ADS

}  // namespace autd::link
