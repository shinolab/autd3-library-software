// File: firmware_version.hpp
// Project: include
// Created Date: 30/03/2020
// Author: Shun Suzuki
// -----
// Last Modified: 08/03/2021
// Modified By: Shun Suzuki (suzuki@hapis.k.u-tokyo.ac.jp)
// -----
// Copyright (c) 2020 Hapis Lab. All rights reserved.
//

#pragma once

#include <iostream>
#include <sstream>
#include <string>

namespace autd {

/**
 * @brief Firmware information
 */
class FirmwareInfo {
 public:
  FirmwareInfo(const uint16_t idx, const uint16_t cpu_ver, const uint16_t fpga_ver)
      : _idx(idx), _cpu_version_number(cpu_ver), _fpga_version_number(fpga_ver) {}

  /**
   * @brief Get cpu firmware version
   */
  [[nodiscard]] std::string cpu_version() const { return firmware_version_map(_cpu_version_number); }
  /**
   * @brief Get fpga firmware version
   */
  [[nodiscard]] std::string fpga_version() const { return firmware_version_map(_fpga_version_number); }

  friend inline std::ostream& operator<<(std::ostream&, const FirmwareInfo&);

 private:
  uint16_t _idx;
  uint16_t _cpu_version_number;
  uint16_t _fpga_version_number;

  static std::string firmware_version_map(const uint16_t version_number) {
    if (version_number == 0) {
      return "older than v0.4";
    } else if (version_number <= 6) {
      std::stringstream ss;
      ss << "v0." << version_number + 3;
      return ss.str();
    } else {
      return "unknown: " + std::to_string(version_number);
    }
  }
};

inline std::ostream& operator<<(std::ostream& os, const FirmwareInfo& obj) {
  os << obj._idx << ": CPU = " << obj.cpu_version() << ", FPGA = " << obj.fpga_version();
  return os;
}
}  // namespace autd
