// File: ethercat_link.cpp
// Project: lib
// Created Date: 01/06/2016
// Author: Seki Inoue
// -----
// Last Modified: 20/02/2020
// Modified By: Shun Suzuki (suzuki@hapis.k.u-tokyo.ac.jp)
// -----
// Copyright (c) 2016-2020 Hapis Lab. All rights reserved.
//

#include "ethercat_link.hpp"

#include <AdsLib.h>
#include <windows.h>

#include <algorithm>
#include <exception>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "privdef.hpp"
// XXX: should be configuarable?
#define INDEX_GROUP (0x3040030)
#define INDEX_OFFSET_BASE (0x81000000)
#define PORT (301)

void autd::internal::EthercatLink::Open(std::string location) {
  auto sep = autd::split(location, ':');

  if (sep.size() == 1) {
    this->Open(sep[0], "");
  } else if (sep.size() == 2) {
    this->Open(sep[1], sep[0]);
  } else {
    throw std::runtime_error("Invalid address");
  }
}
void autd::internal::EthercatLink::Open(std::string ams_net_id, std::string ipv4addr) {
  auto octets = autd::split(ams_net_id, '.');
  if (octets.size() != 6) {
    throw std::runtime_error("Ams net id must have 6 octets");
  }
  if (ipv4addr == "") {
    // Extract ipv6 addr from leading four octets of the ams net id.
    for (int i = 0; i < 3; i++) {
      ipv4addr += octets[i] + ".";
    }
    ipv4addr += octets[3];
  }
  this->_netId = {static_cast<uint8_t>(std::stoi(octets[0])), static_cast<uint8_t>(std::stoi(octets[1])), static_cast<uint8_t>(std::stoi(octets[2])),
                  static_cast<uint8_t>(std::stoi(octets[3])), static_cast<uint8_t>(std::stoi(octets[4])), static_cast<uint8_t>(std::stoi(octets[5]))};
  if (AdsAddRoute(this->_netId, ipv4addr.c_str())) {
    std::cerr << "Error: Could not connect to remote." << std::endl;
    return;
  }
  // open a new ADS port
  this->_port = AdsPortOpenEx();
  if (!this->_port) {
    std::cerr << "Error: Failed to open a new ADS port." << std::endl;
  }
}
void autd::internal::EthercatLink::Close() {
  this->_port = 0;
  AdsPortCloseEx(this->_port);
}
bool autd::internal::EthercatLink::is_open() { return (this->_port > 0); }
void autd::internal::EthercatLink::Send(size_t size, std::unique_ptr<uint8_t[]> buf) {
  const AmsAddr pAddr = {this->_netId, PORT};
  long ret = AdsSyncWriteReqEx(this->_port,  // NOLINT
                               &pAddr, INDEX_GROUP, INDEX_OFFSET_BASE, static_cast<uint32_t>(size), &buf[0]);
  if (ret > 0) {
    switch (ret) {
      case ADSERR_DEVICE_INVALIDSIZE:
        std::cerr << "The number of devices is invalid." << std::endl;
        break;
      default:
        std::cerr << "Error on sending data: " << std::hex << ret << std::endl;
    }
    throw static_cast<int>(ret);
  }
}
bool autd::internal::EthercatLink::CalibrateModulation() {
  return true;  // No need to call CalibrateModulation() for TwinCAT.
}

void autd::internal::EthercatLink::SetWaitForProcessMsg(bool is_wait) {
  return;  // Not implemented or no need...?
}

// for localhost connection
#ifdef _WIN32
typedef long(_stdcall *TcAdsPortOpenEx)(void);                    // NOLINT
typedef long(_stdcall *TcAdsPortCloseEx)(long);                   // NOLINT
typedef long(_stdcall *TcAdsGetLocalAddressEx)(long, AmsAddr *);  // NOLINT
typedef long(_stdcall *TcAdsSyncWriteReqEx)(long, AmsAddr *,      // NOLINT
                                            unsigned long,        // NOLINT
                                            unsigned long,        // NOLINT
                                            unsigned long,        // NOLINT
                                            void *);              // NOLINT
#ifdef _WIN64
#define TCADS_AdsPortOpenEx "AdsPortOpenEx"
#define TCADS_AdsGetLocalAddressEx "AdsGetLocalAddressEx"
#define TCADS_AdsPortCloseEx "AdsPortCloseEx"
#define TCADS_AdsSyncWriteReqEx "AdsSyncWriteReqEx"
#else
#define TCADS_AdsPortOpenEx "_AdsPortOpenEx@0"
#define TCADS_AdsGetLocalAddressEx "_AdsGetLocalAddressEx@8"
#define TCADS_AdsPortCloseEx "_AdsPortCloseEx@4"
#define TCADS_AdsSyncWriteReqEx "_AdsSyncWriteReqEx@24"
#endif

void autd::internal::LocalEthercatLink::Open(std::string location) {
  this->lib = LoadLibrary("TcAdsDll.dll");
  if (lib == nullptr) {
    throw std::runtime_error("couldn't find TcADS-DLL.");
    return;
  }
  // open a new ADS port
  TcAdsPortOpenEx portOpen = (TcAdsPortOpenEx)GetProcAddress(this->lib, TCADS_AdsPortOpenEx);
  this->_port = (*portOpen)();
  if (!this->_port) {
    std::cerr << "Error: Failed to open a new ADS port." << std::endl;
  }
  AmsAddr addr;
  TcAdsGetLocalAddressEx getAddr = (TcAdsGetLocalAddressEx)GetProcAddress(this->lib, TCADS_AdsGetLocalAddressEx);
  long nErr = getAddr(this->_port, &addr);  // NOLINT
  if (nErr) std::cerr << "Error: AdsGetLocalAddress: " << nErr << std::endl;
  this->_netId = addr.netId;
}
void autd::internal::LocalEthercatLink::Close() {
  this->_port = 0;
  TcAdsPortCloseEx portClose = (TcAdsPortCloseEx)GetProcAddress(this->lib, TCADS_AdsPortCloseEx);
  (*portClose)(this->_port);
}
void autd::internal::LocalEthercatLink::Send(size_t size, std::unique_ptr<uint8_t[]> buf) {
  AmsAddr addr = {this->_netId, PORT};
  TcAdsSyncWriteReqEx write = (TcAdsSyncWriteReqEx)GetProcAddress(this->lib, TCADS_AdsSyncWriteReqEx);
  long ret = write(this->_port,  // NOLINT
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
#else
void autd::internal::LocalEthercatLink::Open(std::string location) {
  throw runtime_error(
      "Link to localhost has not been compiled. Rebuild this library on a "
      "Twincat3 host machine with TcADS-DLL.");
}
void autd::internal::LocalEthercatLink::Close() {}
void autd::internal::LocalEthercatLink::Send(size_t size, std::unique_ptr<uint8_t[]> buf) {}
#endif  // TC_ADS
