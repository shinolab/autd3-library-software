// File: linalg.hpp
// Project: include
// Created Date: 20/02/2021
// Author: Shun Suzuki
// -----
// Last Modified: 24/02/2021
// Modified By: Shun Suzuki (suzuki@hapis.k.u-tokyo.ac.jp)
// -----
// Copyright (c) 2021 Hapis Lab. All rights reserved.
//

#pragma once

#ifdef ENABLE_EIGEN
#if WIN32
#include <codeanalysis/warnings.h>  // NOLINT
#pragma warning(push)
#pragma warning(disable : ALL_CODE_ANALYSIS_WARNINGS)
#endif
#include <Eigen/Dense>
#if WIN32
#pragma warning(pop)
#endif
#else
#include "linalg/matrix.hpp"
#include "linalg/quaternion.hpp"
#include "linalg/vector.hpp"
#endif

#include "autd_types.hpp"

namespace autd {
#ifdef ENABLE_EIGEN
using Vector3 = Eigen::Matrix<Float, 3, 1>;
using Quaternion = Eigen::Quaternion<Float>;
inline Quaternion AngleAxis(Float a, Vector3 v) { return Quaternion(Eigen::AngleAxis<Float>(a, v)); }
inline Matrix4x4 Transform(Vector3 v, Quaternion q) {
  const Eigen::Transform<Float, 3, Eigen::Affine> transform_matrix = Eigen::Translation<Float, 3>(v) * q;
  return transform_matrix.matrix();
}
#else
using Vector3 = _utils::Vector3<Float>;
using Matrix4x4 = _utils::Matrix<Float>;
using Quaternion = _utils::Quaternion<Float>;

static Quaternion AngleAxis(Float a, const Vector3& v) { return Quaternion::AngleAxis(a, v); }

inline Matrix4x4 Translation(const Vector3& v, const Quaternion& q) {
  Matrix4x4 transform_matrix;
  transform_matrix(0, 0) = 1.f - 2.f * q.y() * q.y() - 2.f * q.z() * q.z();
  transform_matrix(0, 1) = 2 * q.x() * q.y() + 2 * q.y() * q.z();
  transform_matrix(0, 2) = 2 * q.x() * q.z() - 2 * q.w() * q.y();
  transform_matrix(0, 3) = v.x();
  transform_matrix(1, 0) = 2 * q.x() * q.y() - 2 * q.y() * q.z();
  transform_matrix(1, 1) = 1 - 2 * q.x() * q.x() - 2 * q.z() * q.z();
  transform_matrix(1, 2) = 2 * q.y() * q.z() + 2 * q.w() * q.x();
  transform_matrix(1, 3) = v.y();
  transform_matrix(2, 0) = 2 * q.x() * q.z() + 2 * q.w() * q.y();
  transform_matrix(2, 1) = 2 * q.y() * q.z() - 2 * q.w() * q.x();
  transform_matrix(2, 2) = 1 - 2 * q.x() * q.x() - 2 * q.y() * q.y();
  transform_matrix(2, 3) = v.z();
  transform_matrix(3, 0) = 0;
  transform_matrix(3, 1) = 0;
  transform_matrix(3, 2) = 0;
  transform_matrix(3, 3) = 1;
  return transform_matrix;
}
#endif

}  // namespace autd
