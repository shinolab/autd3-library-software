// File: firmware_version.hpp
// Project: include
// Created Date: 30/03/2020
// Author: Shun Suzuki
// -----
// Last Modified: 01/07/2020
// Modified By: Shun Suzuki (suzuki@hapis.k.u-tokyo.ac.jp)
// -----
// Copyright (c) 2020 Hapis Lab. All rights reserved.
//

#pragma once

#include <iostream>
#include <string>
#include <vector>

namespace autd {

/**
 * @brief Firmware information
 */
class FirmwareInfo {
  friend class AUTDController;

 public:
  /**
   * @brief Get cpu firmware version
   */
  std::string cpu_version() const { return firmware_version_map(_cpu_version_number); }
  /**
   * @brief Get fpga firmware version
   */
  std::string fpga_version() const { return firmware_version_map(_fpga_version_number); }

  friend inline std::ostream& operator<<(std::ostream&, const FirmwareInfo&);

 private:
  FirmwareInfo() : FirmwareInfo(0, 0, 0) {}
  FirmwareInfo(uint16_t idx, uint16_t cpu_ver, uint16_t fpga_ver) {
    _idx = idx;
    _cpu_version_number = cpu_ver;
    _fpga_version_number = fpga_ver;
  }

  uint16_t _idx;
  uint16_t _cpu_version_number;
  uint16_t _fpga_version_number;

  static std::string firmware_version_map(uint16_t version_number) {
    switch (version_number) {
      case 0:
        return "older than v0.4";
      case 1:
        return "v0.4";
      case 2:
        return "v0.5";
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
