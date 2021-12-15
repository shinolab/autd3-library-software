// File: lpf.hpp
// Project: modulation
// Created Date: 10/12/2021
// Author: Shun Suzuki
// -----
// Last Modified: 10/12/2021
// Modified By: Shun Suzuki (suzuki@hapis.k.u-tokyo.ac.jp)
// -----
// Copyright (c) 2021 Hapis Lab. All rights reserved.
//

#pragma once

#include <vector>

#include "autd3/core/modulation.hpp"

namespace autd::modulation {

/**
 * @brief LPF Modulation to reduce noise
 */
class LPF final : public core::Modulation {
 public:
  /**
   * \param modulation Modulation which passes through LPF to reduce noise
   */
  LPF(Modulation& modulation);

  void calc() override;

  ~LPF() override = default;
  LPF(const LPF& v) noexcept = delete;
  LPF& operator=(const LPF& obj) = delete;
  LPF(LPF&& obj) = default;
  LPF& operator=(LPF&& obj) = default;

 private:
  std::vector<uint8_t> _resampled;
};
}  // namespace autd::modulation
