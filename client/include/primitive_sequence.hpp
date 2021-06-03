// File: primitive_sequence.hpp
// Project: include
// Created Date: 14/05/2021
// Author: Shun Suzuki
// -----
// Last Modified: 19/05/2021
// Modified By: Shun Suzuki (suzuki@hapis.k.u-tokyo.ac.jp)
// -----
// Copyright (c) 2021 Hapis Lab. All rights reserved.
//

#pragma once

#include "core/sequence.hpp"

namespace autd::sequence {

using core::SequencePtr;
using PointSequence = core::PointSequence;

/**
 * @brief Utility to generate PointSequence on a circumference.
 */
class Circumference : PointSequence {
 public:
  /**
   * @brief Generate PointSequence with control points on a circumference.
   * @param[in] center Center of the circumference
   * @param[in] normal Normal vector of the circumference
   * @param[in] radius Radius of the circumference
   * @param[in] n Number of the control points
   */
  static SequencePtr create(const core::Vector3& center, const core::Vector3& normal, double radius, size_t n);
};
}  // namespace autd::sequence