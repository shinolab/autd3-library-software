// File: utils.hpp
// Project: include
// Created Date: 16/05/2021
// Author: Shun Suzuki
// -----
// Last Modified: 22/11/2021
// Modified By: Shun Suzuki (suzuki@hapis.k.u-tokyo.ac.jp)
// -----
// Copyright (c) 2021 Hapis Lab. All rights reserved.
//

#pragma once

#include <cmath>

#include "core/geometry.hpp"

namespace autd::utils {
constexpr double DIR_COEFFICIENT_A[] = {1.0, 1.0, 1.0, 0.891250938, 0.707945784, 0.501187234, 0.354813389, 0.251188643, 0.199526231};
constexpr double DIR_COEFFICIENT_B[] = {
    0., 0., -0.00459648054721, -0.0155520765675, -0.0208114779827, -0.0182211227016, -0.0122437497109, -0.00780345575475, -0.00312857467007};
constexpr double DIR_COEFFICIENT_C[] = {
    0., 0., -0.000787968093807, -0.000307591508224, -0.000218348633296, 0.00047738416141, 0.000120353137658, 0.000323676257958, 0.000143850511};
constexpr double DIR_COEFFICIENT_D[] = {
    0., 0., 1.60125528528e-05, 2.9747624976e-06, 2.31910931569e-05, -1.1901034125e-05, 6.77743734332e-06, -5.99548024824e-06, -4.79372835035e-06};

/**
 * \brief Utility class to calculate directivity of ultrasound transducer.
 */
class Directivity {
 public:
  /**
   * \brief Directivity of T4010A1
   * \param theta_deg zenith angle in degree
   * \return directivity
   */
  static double t4010a1(double theta_deg) {
    theta_deg = std::abs(theta_deg);
    while (theta_deg > 90) theta_deg = std::abs(180 - theta_deg);
    const auto i = static_cast<size_t>(std::ceil(theta_deg / 10));
    if (i == 0) return 1;
    const auto a = DIR_COEFFICIENT_A[i - 1];
    const auto b = DIR_COEFFICIENT_B[i - 1];
    const auto c = DIR_COEFFICIENT_C[i - 1];
    const auto d = DIR_COEFFICIENT_D[i - 1];
    const auto x = theta_deg - static_cast<double>(i - 1) * 10;
    return a + b * x + c * x * x + d * x * x * x;
  }
};

inline std::complex<double> transfer(const core::Transducer& transducer, const core::Vector3& trans_norm, const core::Vector3& target_pos,
                                     const double wave_number, const double attenuation) {
  const auto diff = target_pos - transducer.position();
  const auto dist = diff.norm();
  const auto theta = std::atan2(diff.dot(trans_norm), dist * trans_norm.norm()) * 180.0 / M_PI;
  const auto directivity = Directivity::t4010a1(theta);
  return directivity / dist * std::exp(std::complex<double>(-dist * attenuation, -wave_number * dist));
}
}  // namespace autd::utils
