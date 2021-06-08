// File: utils.hpp
// Project: core
// Created Date: 08/06/2021
// Author: Shun Suzuki
// -----
// Last Modified: 08/06/2021
// Modified By: Shun Suzuki (suzuki@hapis.k.u-tokyo.ac.jp)
// -----
// Copyright (c) 2021 Hapis Lab. All rights reserved.
//

#pragma once

namespace autd::core {

class Utilities {
 public:
  /**
   * \brief Convert ultrasound amplitude to duty ratio.
   * \param amp ultrasound amplitude (0 to 1). The amp will be clamped in the range [0, 1].
   * \return duty ratio
   */
  static uint8_t to_duty(const double amp) noexcept {
    const auto d = std::asin(std::clamp(amp, 0.0, 1.0)) / M_PI;  //  duty (0 ~ 0.5)
    return static_cast<uint8_t>(511.0 * d);
  }
};

}  // namespace autd::core
