// File: emulator_link.cpp
// Project: lib
// Created Date: 29/04/2020
// Author: Shun Suzuki
// -----
// Last Modified: 01/07/2020
// Modified By: Shun Suzuki (suzuki@hapis.k.u-tokyo.ac.jp)
// -----
// Copyright (c) 2020 Hapis Lab. All rights reserved.
//

#include "emulator_link.hpp"

#if _WINDOWS
#include <atlstr.h>
#include <winsock2.h>
#include <ws2tcpip.h>
#endif

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

#include "consts.hpp"
#include "privdef.hpp"

namespace autd::link {

LinkPtr EmulatorLink::Create(std::string ipaddr, int32_t port, GeometryPtr geometry) {
  auto link = std::make_shared<EmulatorLink>();
  link->_ipaddr = ipaddr;
  link->_port = port;
  link->_geometry = geometry;

  return link;
}

void EmulatorLink::Open() {
#if _WINDOWS
  WSAData wsaData;
  WSAStartup(MAKEWORD(2, 0), &wsaData);
  _socket = socket(AF_INET, SOCK_DGRAM, 0);
  _addr.sin_family = AF_INET;
  _addr.sin_port = htons(_port);
  CString ipaddr(_ipaddr.c_str());
  inet_pton(AF_INET, ipaddr, &_addr.sin_addr.S_un.S_addr);
#endif
  SetGeometry();
  _is_open = true;
}

void EmulatorLink::Close() {
  if (_is_open) {
    auto buf = std::make_unique<uint8_t[]>(1);
    buf[0] = 0x00;
    Send(1, std::move(buf));
#if _WINDOWS
    closesocket(_socket);
    WSACleanup();
#endif
    _is_open = false;
  }
}

void EmulatorLink::Send(size_t size, std::unique_ptr<uint8_t[]> buf) {
  _last_ms_id = buf[0];
  std::unique_ptr<const uint8_t[]> send_buf = std::move(buf);
#if _WINDOWS
  sendto(_socket, (const char *)(send_buf.get()), static_cast<int>(size), 0, (struct sockaddr *)&_addr, sizeof(_addr));
#endif
}

std::vector<uint8_t> EmulatorLink::Read(uint32_t buffer_len) { return std::vector<uint8_t>(buffer_len, _last_ms_id); }

bool EmulatorLink::is_open() { return _is_open; }

void EmulatorLink::SetGeometry() {
  auto geomrty = this->_geometry;
  const auto vec_size = 3 * sizeof(Vector3) / sizeof(double) * sizeof(float);
  const auto size = geomrty->numDevices() * vec_size + sizeof(float);
  auto buf = std::make_unique<uint8_t[]>(size);
  float header;
  auto uh = reinterpret_cast<uint8_t *>(&header);
  uh[0] = 0xff;
  uh[1] = 0xf0;
  uh[2] = 0xff;
  uh[3] = 0xff;
  {
    auto fbuf = reinterpret_cast<float *>(&buf[0]);
    fbuf[0] = header;
  }
  auto fbuf = reinterpret_cast<float *>(&buf[sizeof(float)]);
  for (size_t i = 0; i < geomrty->numDevices(); i++) {
    auto trans_id = static_cast<int>(i * NUM_TRANS_IN_UNIT);
    auto origin = geomrty->position(trans_id);
    auto right = geomrty->x_direction(trans_id);
    auto up = geomrty->y_direction(trans_id);
    fbuf[9 * i] = static_cast<float>(origin.x());
    fbuf[9 * i + 1] = static_cast<float>(origin.y());
    fbuf[9 * i + 2] = static_cast<float>(origin.z());
    fbuf[9 * i + 3] = static_cast<float>(right.x());
    fbuf[9 * i + 4] = static_cast<float>(right.y());
    fbuf[9 * i + 5] = static_cast<float>(right.z());
    fbuf[9 * i + 6] = static_cast<float>(up.x());
    fbuf[9 * i + 7] = static_cast<float>(up.y());
    fbuf[9 * i + 8] = static_cast<float>(up.z());
  }

  Send(size, std::move(buf));
}
};  // namespace autd::link
