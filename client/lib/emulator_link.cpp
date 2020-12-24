// File: emulator_link.cpp
// Project: lib
// Created Date: 29/04/2020
// Author: Shun Suzuki
// -----
// Last Modified: 25/12/2020
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
#include <iostream>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "consts.hpp"
#include "geometry.hpp"
#include "privdef.hpp"

namespace autd::link {

LinkPtr EmulatorLink::Create(const std::string &ip_addr, const uint16_t port, const GeometryPtr &geometry) {
  LinkPtr link = std::make_unique<EmulatorLink>(ip_addr, port, geometry);
  return link;
}

void EmulatorLink::Open() {
#if _WINDOWS
#pragma warning(push)
#pragma warning(disable : 6031)
  WSAData wsaData{};
  WSAStartup(MAKEWORD(2, 0), &wsaData);
#pragma warning(pop)
  _socket = socket(AF_INET, SOCK_DGRAM, 0);
  _addr.sin_family = AF_INET;
  _addr.sin_port = htons(this->_port);
  const CString ip_addr(_ip_addr.c_str());
  inet_pton(AF_INET, ip_addr, &_addr.sin_addr.S_un.S_addr);
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

void EmulatorLink::Send(const size_t size, std::unique_ptr<uint8_t[]> buf) {
  _last_ms_id = buf[0];
  const std::unique_ptr<const uint8_t[]> send_buf = std::move(buf);
#if _WINDOWS
  sendto(_socket, reinterpret_cast<const char *>(send_buf.get()), static_cast<int>(size), 0, reinterpret_cast<sockaddr *>(&_addr), sizeof(_addr));
#endif
}

std::vector<uint8_t> EmulatorLink::Read(const uint32_t buffer_len) { return std::vector<uint8_t>(buffer_len, _last_ms_id); }

bool EmulatorLink::is_open() { return _is_open; }

void EmulatorLink::SetGeometry() {
  auto geometry = this->_geometry;
  const auto vec_size = 3 * sizeof(Vector3) / sizeof(double) * sizeof(float);
  const auto size = geometry->numDevices() * vec_size + sizeof(float);
  auto buf = std::make_unique<uint8_t[]>(size);
  float header;
  const auto uh = reinterpret_cast<uint8_t *>(&header);
  uh[0] = 0xff;
  uh[1] = 0xf0;
  uh[2] = 0xff;
  uh[3] = 0xff;
  {
    const auto float_buf = reinterpret_cast<float *>(&buf[0]);
    float_buf[0] = header;
  }
  const auto float_buf = reinterpret_cast<float *>(&buf[sizeof(float)]);
  for (size_t i = 0; i < geometry->numDevices(); i++) {
    const auto trans_id = i * NUM_TRANS_IN_UNIT;
    auto origin = geometry->position(trans_id);
    auto right = geometry->x_direction(trans_id);
    auto up = geometry->y_direction(trans_id);
    float_buf[9 * i] = static_cast<float>(origin.x());
    float_buf[9 * i + 1] = static_cast<float>(origin.y());
    float_buf[9 * i + 2] = static_cast<float>(origin.z());
    float_buf[9 * i + 3] = static_cast<float>(right.x());
    float_buf[9 * i + 4] = static_cast<float>(right.y());
    float_buf[9 * i + 5] = static_cast<float>(right.z());
    float_buf[9 * i + 6] = static_cast<float>(up.x());
    float_buf[9 * i + 7] = static_cast<float>(up.y());
    float_buf[9 * i + 8] = static_cast<float>(up.z());
  }

  Send(size, std::move(buf));
}
};  // namespace autd::link
