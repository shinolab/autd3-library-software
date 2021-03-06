// File: linalg.hpp
// Project: include
// Created Date: 20/02/2021
// Author: Shun Suzuki
// -----
// Last Modified: 06/03/2021
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
using Vector4 = Eigen::Matrix<Float, 4, 1>;
using Matrix4x4 = Eigen::Matrix<Float, 4, 4>;
using Quaternion = Eigen::Quaternion<Float>;
inline Quaternion AngleAxis(const Float a, const Vector3& v) { return Quaternion(Eigen::AngleAxis<Float>(a, v)); }
inline Matrix4x4 Translation(const Vector3& v, const Quaternion& q) {
  const Eigen::Transform<Float, 3, Eigen::Affine> transform_matrix = Eigen::Translation<Float, 3>(v) * q;
  return transform_matrix.matrix();
}
#else
using Vector3 = _utils::Vector3<Float>;
using Vector4 = _utils::Vector4<Float>;
using Matrix4x4 = _utils::Matrix4x4<Float>;
using Quaternion = _utils::Quaternion<Float>;

inline Quaternion AngleAxis(const Float a, const Vector3& v) { return Quaternion::AngleAxis(a, v); }
inline Matrix4x4 Translation(const Vector3& v, const Quaternion& q) { return Matrix4x4::Translation(v, q); }
#endif
template <typename V>
Vector3 ToVector3(const V& v) {
  Vector3 r;
  r(0) = v(0);
  r(1) = v(1);
  r(2) = v(2);
  return r;
}
}  // namespace autd
