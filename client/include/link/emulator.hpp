// File: emulator_link.hpp
// Project: lib
// Created Date: 29/04/2020
// Author: Shun Suzuki
// -----
// Last Modified: 03/04/2021
// Modified By: Shun Suzuki (suzuki@hapis.k.u-tokyo.ac.jp)
// -----
// Copyright (c) 2020 Hapis Lab. All rights reserved.
//

#pragma once

#if _WINDOWS
#define NOMINMAX
#include <WinSock2.h>
#endif

#include <memory>
#include <string>
#include <utility>

#include "geometry.hpp"
#include "link.hpp"
#include "link/emulator.hpp"

namespace autd::link {
/**
 * @brief Experimental: Link to connect with [Emulator](https://github.com/shinolab/autd-emulator)
 */
class EmulatorLink final : public Link {
 public:
  static LinkPtr Create(const std::string& ip_addr, uint16_t port, const GeometryPtr& geometry);
  EmulatorLink(std::string ip_addr, const uint16_t port, GeometryPtr geometry)
      : Link(), _ip_addr(std::move(ip_addr)), _geometry(std::move(geometry)), _port(port) {}
  ~EmulatorLink() override = default;
  EmulatorLink(const EmulatorLink& v) noexcept = delete;
  EmulatorLink& operator=(const EmulatorLink& obj) = delete;
  EmulatorLink(EmulatorLink&& obj) = delete;
  EmulatorLink& operator=(EmulatorLink&& obj) = delete;

  bool Open() override;
  bool Close() override;
  std::optional<std::string> Send(size_t size, std::unique_ptr<uint8_t[]> buf) override;
  std::optional<std::string> Read(uint8_t* rx, uint32_t buffer_len) override;
  bool is_open() override;
  void SetGeometry();

 private:
  bool _is_open = false;
  std::string _ip_addr;
  GeometryPtr _geometry;
  uint16_t _port = 0;
#if _WINDOWS
  SOCKET _socket = {};
  sockaddr_in _addr = {};
#endif
  uint8_t _last_msg_id = 0;
};
}  // namespace autd::link
