// File: modulation.hpp
// Project: include
// Created Date: 04/11/2018
// Author: Shun Suzuki
// -----
// Last Modified: 20/02/2021
// Modified By: Shun Suzuki (suzuki@hapis.k.u-tokyo.ac.jp)
// -----
// Copyright (c) 2018-2020 Hapis Lab. All rights reserved.
//

#pragma once

#define _USE_MATH_DEFINES  // NOLINT
#include <math.h>

#include <limits>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "autd_types.hpp"
#include "configuration.hpp"

namespace autd {

namespace modulation {
class Modulation;
}

using ModulationPtr = std::shared_ptr<modulation::Modulation>;

namespace modulation {

inline double Sinc(const double x) noexcept {
  if (fabs(x) < std::numeric_limits<double>::epsilon()) return 1;
  return sin(M_PI * x) / (M_PI * x);
}

/**
 * @brief Modulation controls the amplitude modulation
 */
class Modulation {
 public:
  Modulation() noexcept;
  virtual ~Modulation() = default;
  Modulation(const Modulation& v) noexcept = default;
  Modulation& operator=(const Modulation& obj) = default;
  Modulation(Modulation&& obj) = default;
  Modulation& operator=(Modulation&& obj) = default;

  /**
   * @brief Generate empty modulation, which produce static pressure
   */
  static ModulationPtr Create(uint8_t amp = 0xff);
  virtual void Build(Configuration config);
  std::vector<uint8_t> buffer;
  size_t& sent();

 private:
  size_t _sent;
};
}  // namespace modulation
}  // namespace autd
