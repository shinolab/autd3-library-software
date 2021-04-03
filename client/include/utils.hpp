// File: utils.hpp
// Project: include
// Created Date: 06/03/2021
// Author: Shun Suzuki
// -----
// Last Modified: 03/04/2021
// Modified By: Shun Suzuki (suzuki@hapis.k.u-tokyo.ac.jp)
// -----
// Copyright (c) 2021 Hapis Lab. All rights reserved.
//

#pragma once

namespace autd::utils {

#ifdef USE_DOUBLE_AUTD
constexpr double DIR_COEFF_A[] = {1.0, 1.0, 1.0, 0.891250938, 0.707945784, 0.501187234, 0.354813389, 0.251188643, 0.199526231};
constexpr double DIR_COEFF_B[] = {
    0., 0., -0.00459648054721, -0.0155520765675, -0.0208114779827, -0.0182211227016, -0.0122437497109, -0.00780345575475, -0.00312857467007};
constexpr double DIR_COEFF_C[] = {
    0., 0., -0.000787968093807, -0.000307591508224, -0.000218348633296, 0.00047738416141, 0.000120353137658, 0.000323676257958, 0.000143850511};
constexpr double DIR_COEFF_D[] = {
    0., 0., 1.60125528528e-05, 2.9747624976e-06, 2.31910931569e-05, -1.1901034125e-05, 6.77743734332e-06, -5.99548024824e-06, -4.79372835035e-06};
#else
constexpr float DIR_COEFF_A[] = {1.0f, 1.0f, 1.0f, 0.891250938f, 0.707945784f, 0.501187234f, 0.354813389f, 0.251188643f, 0.199526231f};
constexpr float DIR_COEFF_B[] = {
    0.f, 0.f, -0.00459648054721f, -0.0155520765675f, -0.0208114779827f, -0.0182211227016f, -0.0122437497109f, -0.00780345575475f, -0.00312857467007f};
constexpr float DIR_COEFF_C[] = {0.f,
                                 0.f,
                                 -0.000787968093807f,
                                 -0.000307591508224f,
                                 -0.000218348633296f,
                                 0.00047738416141f,
                                 0.000120353137658f,
                                 0.000323676257958f,
                                 0.000143850511f};
constexpr float DIR_COEFF_D[] = {0.f,
                                 0.f,
                                 1.60125528528e-05f,
                                 2.9747624976e-06f,
                                 2.31910931569e-05f,
                                 -1.1901034125e-05f,
                                 6.77743734332e-06f,
                                 -5.99548024824e-06f,
                                 -4.79372835035e-06f};
#endif

static inline Float directivityT4010A1(Float theta_deg) {
  theta_deg = abs(theta_deg);

  while (theta_deg > 90) {
    theta_deg = abs(180 - theta_deg);
  }

  const auto i = static_cast<size_t>(ceil(theta_deg / 10));

  if (i == 0) {
    return 1;
  }

  const auto a = DIR_COEFF_A[i - 1];
  const auto b = DIR_COEFF_B[i - 1];
  const auto c = DIR_COEFF_C[i - 1];
  const auto d = DIR_COEFF_D[i - 1];
  const auto x = theta_deg - static_cast<Float>(i - 1) * 10;
  return a + b * x + c * x * x + d * x * x * x;
}
}  // namespace autd::utils
