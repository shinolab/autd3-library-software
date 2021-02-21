// File: primitive.cpp
// Project: primitive
// Created Date: 20/02/2021
// Author: Shun Suzuki
// -----
// Last Modified: 21/02/2021
// Modified By: Shun Suzuki (suzuki@hapis.k.u-tokyo.ac.jp)
// -----
// Copyright (c) 2021 Hapis Lab. All rights reserved.
//

#include "sequence/primitive.hpp"

namespace autd::sequence {

static Vector3 GetOrthogonal(const Vector3& v) {
  const auto a = Vector3::UnitX();
  if (acos(v.dot(a)) < PI / 2) {
    const auto b = Vector3::UnitY();
    return v.cross(b);
  }

  return v.cross(a);
}

SequencePtr CreateImpl(const Vector3& center, const Vector3& normal, const Float radius, const size_t n) {
  const auto normal_ = normal.normalized();
  const auto n1 = GetOrthogonal(normal_).normalized();
  const auto n2 = normal_.cross(n1).normalized();

  std::vector<Vector3> control_points;
  for (size_t i = 0; i < n; i++) {
    const auto theta = 2 * PI / static_cast<Float>(n) * static_cast<Float>(i);
    auto x = n1 * radius * cos(theta);
    auto y = n2 * radius * sin(theta);
    control_points.emplace_back(center + x + y);
  }
  return PointSequence::Create(control_points);
}

SequencePtr CircumSeq::Create(const Vector3& center, const Vector3& normal, const Float radius, const size_t n) {
  return CreateImpl(center, normal, radius, n);
}
}  // namespace autd::sequence