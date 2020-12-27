// File: consts.hpp
// Project: lib
// Created Date: 07/06/2016
// Author: Seki Inoue
// -----
// Last Modified: 27/12/2020
// Modified By: Shun Suzuki (suzuki@hapis.k.u-tokyo.ac.jp)
// -----
// Copyright (c) 2016-2020 Hapis Lab. All rights reserved.
//

#pragma once

#define _USE_MATH_DEFINES  // NOLINT
#include <math.h>

#include "autd_types.hpp"

namespace autd {
constexpr size_t NUM_TRANS_IN_UNIT = 249;
constexpr size_t NUM_TRANS_X = 18;
constexpr size_t NUM_TRANS_Y = 14;
constexpr Float TRANS_SIZE_MM = ToFloat(10.16);
constexpr Float AUTD_WIDTH = ToFloat(192.0);
constexpr Float AUTD_HEIGHT = ToFloat(151.4);

constexpr Float PI = ToFloat(M_PI);

template <typename T>
constexpr auto IsMissingTransducer(T x, T y) {
  return y == 1 && (x == 1 || x == 2 || x == 16);
}

constexpr auto FPGA_CLOCK = 25600000;

constexpr auto ULTRASOUND_FREQUENCY = 40000;

constexpr auto POINT_SEQ_BUFFER_SIZE_MAX = 40000;
constexpr auto POINT_SEQ_CLK_IDX_MAX = 40000;
constexpr Float POINT_SEQ_BASE_FREQ = ToFloat(40000.0);

using AUTDDataArray = std::array<uint16_t, NUM_TRANS_IN_UNIT>;
}  // namespace autd
