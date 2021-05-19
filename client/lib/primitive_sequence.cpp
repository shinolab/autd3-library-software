// File: primitive_sequence.cpp
// Project: lib
// Created Date: 16/05/2021
// Author: Shun Suzuki
// -----
// Last Modified: 19/05/2021
// Modified By: Shun Suzuki (suzuki@hapis.k.u-tokyo.ac.jp)
// -----
// Copyright (c) 2021 Hapis Lab. All rights reserved.
//

#include "primitive_sequence.hpp"

#include "core/sequence.hpp"

namespace autd::sequence {

SequencePtr Circumference::create(const core::Vector3& center, const core::Vector3& normal, const double radius, const size_t n) {
  auto get_orthogonal = [](const core::Vector3& v) {
    const auto a = core::Vector3::UnitX();
    if (std::acos(v.dot(a)) < M_PI / 2) return v.cross(core::Vector3::UnitY());
    return v.cross(a);
  };

  const auto normal_ = normal.normalized();
  const auto n1 = get_orthogonal(normal_).normalized();
  const auto n2 = normal_.cross(n1).normalized();

  std::vector<core::Vector3> control_points;
  for (size_t i = 0; i < n; i++) {
    const auto theta = 2 * M_PI / static_cast<double>(n) * static_cast<double>(i);
    auto x = n1 * radius * std::cos(theta);
    auto y = n2 * radius * std::sin(theta);
    control_points.emplace_back(center + x + y);
  }
  return PointSequence::create(control_points);
}
}  // namespace autd::sequence
