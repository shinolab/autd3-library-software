// File: emulator_link.hpp
// Project: lib
// Created Date: 29/04/2020
// Author: Shun Suzuki
// -----
// Last Modified: 21/05/2020
// Modified By: Shun Suzuki (suzuki@hapis.k.u-tokyo.ac.jp)
// -----
// Copyright (c) 2020 Hapis Lab. All rights reserved.
//

#pragma once

#if _WINDOWS
#define NOMINMAX
#include <winsock2.h>
#endif

#include <memory>
#include <string>
#include <vector>

#include "emulator_link.hpp"
#include "geometry.hpp"
#include "link.hpp"

namespace autd {
/**
 * @brief Experimental: Link to connect with [Emulator](https://github.com/shinolab/autd-emulator)
 */
class EmulatorLink : public Link {
 public:
  static LinkPtr Create(std::string ipaddr, int32_t port, GeometryPtr geometry);
  ~EmulatorLink() override{};

 protected:
  void Open() final;
  void Close() final;
  void Send(size_t size, std::unique_ptr<uint8_t[]> buf) final;
  std::vector<uint8_t> Read(uint32_t buffer_len) final;
  bool is_open() final;
  bool CalibrateModulation() final;
  void SetGeometry();

 private:
  bool _is_open = false;
  size_t _dev_num = 0;
  std::string _ipaddr;
  GeometryPtr _geometry;
  int32_t _port;
#if _WINDOWS
  SOCKET _socket = {};
  sockaddr_in _addr = {};
#endif
  uint8_t _last_ms_id = 0;
};
}  // namespace autd
