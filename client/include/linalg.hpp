// File: linalg.hpp
// Project: include
// Created Date: 20/02/2021
// Author: Shun Suzuki
// -----
// Last Modified: 30/04/2021
// Modified By: Shun Suzuki (suzuki@hapis.k.u-tokyo.ac.jp)
// -----
// Copyright (c) 2021 Hapis Lab. All rights reserved.
//

#pragma once

#ifndef DISABLE_EIGEN
#if WIN32
#pragma warning(push)
#pragma warning(disable : 26450 26495 26812)
#endif
#ifdef linux
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wmaybe-uninitialized"
#endif
#include <Eigen/Dense>
#if WIN32
#pragma warning(pop)
#endif
#ifdef linux
#pragma GCC diagnostic pop
#endif
#else
#include "linalg/matrix.hpp"
#include "linalg/quaternion.hpp"
#include "linalg/vector.hpp"
#endif

#include "autd_types.hpp"

namespace autd {
#ifndef DISABLE_EIGEN
using Vector3 = Eigen::Matrix<Float, 3, 1>;
using Vector4 = Eigen::Matrix<Float, 4, 1>;
using Matrix4X4 = Eigen::Matrix<Float, 4, 4>;
using Quaternion = Eigen::Quaternion<Float>;
inline Quaternion AngleAxis(const Float a, const Vector3& v) { return Quaternion(Eigen::AngleAxis<Float>(a, v)); }
inline Matrix4X4 Translation(const Vector3& v, const Quaternion& q) {
  const Eigen::Transform<Float, 3, Eigen::Affine> transform_matrix = Eigen::Translation<Float, 3>(v) * q;
  return transform_matrix.matrix();
}
#else
using Vector3 = utils::Vector3<Float>;
using Vector4 = utils::Vector4<Float>;
using Matrix4X4 = utils::Matrix4X4<Float>;
using Quaternion = utils::Quaternion<Float>;

inline Quaternion AngleAxis(const Float a, const Vector3& v) { return Quaternion::AngleAxis(a, v); }
inline Matrix4X4 Translation(const Vector3& v, const Quaternion& q) { return Matrix4X4::Translation(v, q); }
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
