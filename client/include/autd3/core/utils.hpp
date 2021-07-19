// File: utils.hpp
// Project: core
// Created Date: 08/06/2021
// Author: Shun Suzuki
// -----
// Last Modified: 19/07/2021
// Modified By: Shun Suzuki (suzuki@hapis.k.u-tokyo.ac.jp)
// -----
// Copyright (c) 2021 Hapis Lab. All rights reserved.
//

#pragma once

#include <algorithm>
#include <cmath>

namespace autd::core {

class Utilities {
 public:
  /**
   * \brief Convert ultrasound amplitude to duty ratio.
   * \param amp ultrasound amplitude (0 to 1). The amp will be clamped in the range [0, 1].
   * \return duty ratio
   */
  inline static uint8_t to_duty(const double amp) noexcept {
    const auto d = std::asin(std::clamp(amp, 0.0, 1.0)) / M_PI;  //  duty (0 ~ 0.5)
    return static_cast<uint8_t>(510.0 * d);
  }

  /**
   * \brief Convert normalized phase to descrete phase.
   * \param phase nomalized phase
   * \return descrete phase
   */
  inline static uint8_t to_phase(const double phase) noexcept {
    const int d_phase = static_cast<int>(std::round(phase * 256.0)) & 0xFF;
#ifdef AUTD_PHASE_INVERTED
    return static_cast<uint8_t>(0xFF - d_phase);
#else
    return static_cast<uint8_t>(d_phase);
#endif
  }

  /**
   * \brief Pack two uint8_t value to uint16_t
   * \param high high byte
   * \param low high byte
   * \return packed uint16_t value
   */
  inline static uint16_t pack_to_u16(const uint8_t high, const uint8_t low) noexcept {
    auto res = static_cast<uint16_t>(low) & 0x00FF;
    res |= (static_cast<uint16_t>(high) << 8) & 0xFF00;
    return res;
  }
};
}  // namespace autd::core
