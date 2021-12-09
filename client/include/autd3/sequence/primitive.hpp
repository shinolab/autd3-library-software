// File: primitive_sequence.hpp
// Project: include
// Created Date: 14/05/2021
// Author: Shun Suzuki
// -----
// Last Modified: 09/12/2021
// Modified By: Shun Suzuki (suzuki@hapis.k.u-tokyo.ac.jp)
// -----
// Copyright (c) 2021 Hapis Lab. All rights reserved.
//

#pragma once

#include "autd3/core/sequence.hpp"

namespace autd::sequence {

using PointSequence = core::PointSequence;
using GainSequence = core::GainSequence;
using GAIN_MODE = core::GAIN_MODE;

/**
 * @brief Utility to generate PointSequence on a circumference.
 */
class Circumference final : PointSequence {
 public:
  /**
   * @param[in] center Center of the circumference
   * @param[in] normal Normal vector of the circumference
   * @param[in] radius Radius of the circumference
   * @param[in] n Number of the control points
   * @param[in] duty duty of control points
   */
  Circumference(const core::Vector3& center, const core::Vector3& normal, double radius, size_t n, uint8_t duty = 0xFF);
};
}  // namespace autd::sequence
