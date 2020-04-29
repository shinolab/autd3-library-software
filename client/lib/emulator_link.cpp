// File: emulator_link.cpp
// Project: lib
// Created Date: 29/04/2020
// Author: Shun Suzuki
// -----
// Last Modified: 29/04/2020
// Modified By: Shun Suzuki (suzuki@hapis.k.u-tokyo.ac.jp)
// -----
// Copyright (c) 2020 Hapis Lab. All rights reserved.
//

#include "emulator_link.hpp"

#include <atlstr.h>
#include <winsock2.h>
#include <ws2tcpip.h>

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

#include "privdef.hpp"

namespace autd {

void internal::EmulatorLink::Open(std::string location) {
  WSAData wsaData;
  auto tmp = autd::split(location, ':');
  WSAStartup(MAKEWORD(2, 0), &wsaData);
  _socket = socket(AF_INET, SOCK_DGRAM, 0);
  _addr.sin_family = AF_INET;
  _addr.sin_port = htons(stoi(tmp[1]));
  CString ipaddr(tmp[0].c_str());
  inet_pton(AF_INET, ipaddr, &_addr.sin_addr.S_un.S_addr);
  _is_open = true;
}

void internal::EmulatorLink::Close() {
  if (_is_open) {
    closesocket(_socket);
    WSACleanup();
    _is_open = false;
  }
}

void internal::EmulatorLink::Send(size_t size, std::unique_ptr<uint8_t[]> buf) {
  _last_ms_id = buf[0];
  std::unique_ptr<const uint8_t[]> send_buf = std::move(buf);
  sendto(_socket, (const char *)(send_buf.get()), static_cast<int>(size), 0, (struct sockaddr *)&_addr, sizeof(_addr));
}

std::vector<uint8_t> internal::EmulatorLink::Read(uint32_t buffer_len) { return std::vector<uint8_t>(buffer_len, _last_ms_id); }

bool internal::EmulatorLink::is_open() { return _is_open; }

bool internal::EmulatorLink::CalibrateModulation() { return true; }

void internal::EmulatorLink::SetGeometry(GeometryPtr geomrty) {
  const auto vec_size = sizeof(float) * 3 * 3;
  auto size = geomrty->numDevices() * vec_size + sizeof(float);
  auto buf = std::make_unique<uint8_t[]>(size);
  float header;
  ((uint8_t *)(&header))[0] = 0xff;
  ((uint8_t *)(&header))[1] = 0xf0;
  ((uint8_t *)(&header))[2] = 0xff;
  ((uint8_t *)(&header))[3] = 0xff;
  {
    auto fbuf = (float *)(&buf[0]);
    fbuf[0] = header;
  }
  auto fbuf = (float *)(&buf[sizeof(float)]);
   for (size_t i = 0; i < geomrty->numDevices(); i++) {
    auto trans_id = static_cast<int>(i * NUM_TRANS_IN_UNIT);
    auto origin = geomrty->position(trans_id);
    auto right = geomrty->x_direction(trans_id);
    auto up = geomrty->y_direction(trans_id);
    fbuf[vec_size * i] = static_cast<float>(origin.x());
    fbuf[vec_size * i + 1] = static_cast<float>(origin.y());
    fbuf[vec_size * i + 2] = static_cast<float>(origin.z());
    fbuf[vec_size * i + 3] = static_cast<float>(right.x());
    fbuf[vec_size * i + 4] = static_cast<float>(right.y());
    fbuf[vec_size * i + 5] = static_cast<float>(right.z());
    fbuf[vec_size * i + 6] = static_cast<float>(up.x());
    fbuf[vec_size * i + 7] = static_cast<float>(up.y());
    fbuf[vec_size * i + 8] = static_cast<float>(up.z());
  }
   Send(size, std::move(buf));
}
};  // namespace autd