// File: emulator_link.hpp
// Project: lib
// Created Date: 29/04/2020
// Author: Shun Suzuki
// -----
// Last Modified: 29/04/2020
// Modified By: Shun Suzuki (suzuki@hapis.k.u-tokyo.ac.jp)
// -----
// Copyright (c) 2020 Hapis Lab. All rights reserved.
//

#pragma once

#define NOMINMAX
#include <winsock2.h>

#include <memory>
#include <string>
#include <vector>

#include "emulator_link.hpp"
#include "geometry.hpp"
#include "link.hpp"

namespace autd {
namespace internal {
class EmulatorLink : public Link {
 public:
  void Open(std::string location) final;
  void Close() final;
  void Send(size_t size, std::unique_ptr<uint8_t[]> buf) final;
  std::vector<uint8_t> Read(uint32_t buffer_len) final;
  bool is_open() final;
  bool CalibrateModulation() final;
  void SetGeometry(GeometryPtr geometry);

 private:
  bool _is_open = false;
  size_t _dev_num = 0;
  std::string _ifname;
  SOCKET _socket;
  sockaddr_in _addr;
};
}  // namespace internal
}  // namespace autd
