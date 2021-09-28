// File: firmware_version.hpp
// Project: core
// Created Date: 14/04/2021
// Author: Shun Suzuki
// -----
// Last Modified: 28/09/2021
// Modified By: Shun Suzuki (suzuki@hapis.k.u-tokyo.ac.jp)
// -----
// Copyright (c) 2021 Hapis Lab. All rights reserved.
//

#pragma once

#include <sstream>
#include <string>

namespace autd {

/**
 * \brief Firmware information
 */
class FirmwareInfo {
 public:
  FirmwareInfo(const uint16_t idx, const uint16_t cpu_ver, const uint16_t fpga_ver)
      : _idx(idx), _cpu_version_number(cpu_ver), _fpga_version_number(fpga_ver) {}

  /**
   * \brief Get cpu firmware version
   */
  [[nodiscard]] std::string cpu_version() const { return firmware_version_map(_cpu_version_number); }
  /**
   * \brief Get fpga firmware version
   */
  [[nodiscard]] std::string fpga_version() const { return firmware_version_map(_fpga_version_number); }

  friend inline std::ostream& operator<<(std::ostream&, const FirmwareInfo&);

 private:
  uint16_t _idx;
  uint16_t _cpu_version_number;
  uint16_t _fpga_version_number;

  static std::string firmware_version_map(const uint16_t version_number) {
    std::stringstream ss;
    if (version_number == 0) {
      return "older than v0.4";
    }
    if (version_number <= 6) {
      ss << "v0." << version_number + 3;
      return ss.str();
    }
    if (0x000A <= version_number && version_number <= 0x0012) {
      ss << "v1." << version_number - 0x000A;
      return ss.str();
    }
    if (version_number == 0xFFFF) {
      return "emulator";
    }
    if ((version_number & 0xF000) == 0xF000) {
      ss << "v0." << version_number - 0xF000 + 1 << "-freq-shift";
      return ss.str();
    }

    ss << "unknown: " << version_number;
    return ss.str();
  }
};

inline std::ostream& operator<<(std::ostream& os, const FirmwareInfo& obj) {
  os << obj._idx << ": CPU = " << obj.cpu_version() << ", FPGA = " << obj.fpga_version();
  return os;
}
}  // namespace autd
