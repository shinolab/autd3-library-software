// File: sequence.hpp
// Project: include
// Created Date: 01/07/2020
// Author: Shun Suzuki
// -----
// Last Modified: 21/02/2021
// Modified By: Shun Suzuki (suzuki@hapis.k.u-tokyo.ac.jp)
// -----
// Copyright (c) 2020 Hapis Lab. All rights reserved.
//

#pragma once

#include <memory>
#include <vector>

#include "autd_types.hpp"
#include "consts.hpp"
#include "linalg.hpp"
#include "sequence.hpp"

namespace autd::sequence {
/**
 * @brief Utility to generate PointSequence on a circumference.
 */
class CircumSeq : PointSequence {
 public:
  /**
   * @brief Generate PointSequence with control points on a circumference.
   * @param[in] center Center of the circumference
   * @param[in] normal Normal vector of the circumference
   * @param[in] radius Radius of the circumference
   * @param[in] n Number of the control points
   */
  static SequencePtr Create(const Vector3& center, const Vector3& normal, Float radius, size_t n);
};
}  // namespace autd::sequence
