// File: privdef.hpp
// Project: lib
// Created Date: 07/06/2016
// Author: Seki Inoue
// -----
// Last Modified: 30/04/2020
// Modified By: Shun Suzuki (suzuki@hapis.k.u-tokyo.ac.jp)
// -----
// Copyright (c) 2016-2020 Hapis Lab. All rights reserved.
//

#pragma once

#include <stdio.h>

namespace autd {
constexpr size_t NUM_TRANS_IN_UNIT = 249;
constexpr size_t NUM_TRANS_X = 18;
constexpr size_t NUM_TRANS_Y = 14;
constexpr double TRANS_SIZE_MM = 10.18;
template <typename T>
constexpr auto IS_MISSING_TRANSDUCER(T X, T Y) {
  return (Y == 1 && (X == 1 || X == 2 || X == 16));
}

constexpr auto FPGA_CLOCK = 25600000;

constexpr auto ULTRASOUND_FREQUENCY = 40000;
constexpr auto ULTRASOUND_WAVELENGTH = 8.5;

constexpr auto MOD_SAMPLING_FREQ = 4000;
constexpr auto MOD_BUF_SIZE = 4000;
}  // namespace autd
