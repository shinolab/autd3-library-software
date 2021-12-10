// File: utils.hpp
// Project: core
// Created Date: 08/06/2021
// Author: Shun Suzuki
// -----
// Last Modified: 22/11/2021
// Modified By: Shun Suzuki (suzuki@hapis.k.u-tokyo.ac.jp)
// -----
// Copyright (c) 2021 Hapis Lab. All rights reserved.
//

#pragma once

#include <algorithm>
#include <cmath>

#include "autd3/core/hardware_defined.hpp"

namespace autd::core::utils {

/**
 * \brief Convert ultrasound amplitude to duty ratio.
 * \param amp ultrasound amplitude (0 to 1). The amp will be clamped in the range [0, 1].
 * \return duty ratio
 */
static uint8_t to_duty(const double amp) noexcept {
  const auto d = std::asin(std::clamp(amp, 0.0, 1.0)) / M_PI;  //  duty (0 ~ 0.5)
  return static_cast<uint8_t>(510.0 * d);
}

/**
 * \brief Convert phase in radian to discrete phase used in device.
 * \param phase phase in radian
 * \return discrete phase
 */
static uint8_t to_phase(const double phase) noexcept {
  const auto d_phase = static_cast<uint8_t>(static_cast<int>(std::round((phase / (2.0 * M_PI) + 0.5) * 256.0)) & 0xFF);
  return PHASE_INVERTED ? d_phase : 0xFF - d_phase;
}

/**
 * \brief Pack two uint8_t value to uint16_t
 * \param high high byte
 * \param low high byte
 * \return packed uint16_t value
 */
static uint16_t pack_to_u16(const uint8_t high, const uint8_t low) noexcept {
  uint16_t res = static_cast<uint16_t>(low) & 0x00FF;
  res |= static_cast<uint16_t>(high) << 8 & 0xFF00;
  return res;
}

/**
 * \brief Positive modulo
 * \param i left operand
 * \param n right operand
 * \return i % n
 */
static size_t modulo_positive(const int32_t i, const size_t n) {
  int32_t res = i % static_cast<int32_t>(n);
  if (res < 0) res += static_cast<int32_t>(n);
  return static_cast<size_t>(res);
}

}  // namespace autd::core::utils
