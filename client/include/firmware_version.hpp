// File: firmware_version.hpp
// Project: include
// Created Date: 30/03/2020
// Author: Shun Suzuki
// -----
// Last Modified: 30/03/2020
// Modified By: Shun Suzuki (suzuki@hapis.k.u-tokyo.ac.jp)
// -----
// Copyright (c) 2020 Hapis Lab. All rights reserved.
//

#pragma once

#include <iostream>
#include <string>
#include <vector>

namespace autd {

class FirmwareInfo;

#if DLL_FOR_CAPI
using FirmwareInfoList = FirmwareInfo*;
#else
using FirmwareInfoList = std::vector<FirmwareInfo>;
#endif

class FirmwareInfo {
 public:
  FirmwareInfo(uint16_t idx, uint16_t cpu_ver, uint16_t fpga_ver) {
    _idx = idx;
    _cpu_version_number = cpu_ver;
    _fpga_version_number = fpga_ver;
  }

  std::string cpu_version() const { return firmware_version_map(_cpu_version_number); }
  std::string fpga_version() const { return firmware_version_map(_fpga_version_number); }

  friend inline std::ostream& operator<<(std::ostream&, const FirmwareInfo&);

 private:
  uint16_t _idx;
  uint16_t _cpu_version_number;
  uint16_t _fpga_version_number;

  static std::string firmware_version_map(uint16_t version_number) {
    switch (version_number) {
      case 0:
        return "older than v0.4";
      case 1:
        return "v0.4";
      default:
        return "unknown: " + std::to_string(version_number);
    }
  }
};

inline std::ostream& operator<<(std::ostream& os, const FirmwareInfo& obj) {
  os << obj._idx << ": CPU = " << obj.cpu_version() << ", FPGA = " << obj.fpga_version();
  return os;
}
}  // namespace autd
