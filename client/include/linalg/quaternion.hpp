// File: quaternion.hpp
// Project: include
// Created Date: 27/02/2020
// Author: Shun Suzuki
// -----
// Last Modified: 24/02/2021
// Modified By: Shun Suzuki (suzuki@hapis.k.u-tokyo.ac.jp)
// -----
// Copyright (c) 2020 Hapis Lab. All rights reserved.
//

#pragma once

#include <cmath>

#include "vector.hpp"

namespace autd::_utils {
/**
 * @brief Simple quaternion class
 */
template <typename T>
class Quaternion {
 private:
  Vector3<T> _v;
  T _w;

 public:
  Quaternion() noexcept = default;
  Quaternion(T w, T x, T y, T z) {
    this->_v = Vector3(x, y, z);
    this->_w = w;
  }
  Quaternion(const Quaternion& v) noexcept = default;
  Quaternion& operator=(const Quaternion& obj) = default;

  Vector3<T> v() { return _v; }
  T& x() noexcept { return _v.x(); }
  T& y() noexcept { return _v.y(); }
  T& z() noexcept { return _v.z(); }
  T& w() noexcept { return _w; }
  const T& x() const noexcept { return _v.x(); }
  const T& y() const noexcept { return _v.y(); }
  const T& z() const noexcept { return _v.z(); }
  const T& w() const noexcept { return _w; }

  T l2_norm_squared() const { return _v.l2_norm_squared() + _w * _w; }
  T l2_norm() const { return std::sqrt(l2_norm_squared()); }
  T norm() const { return l2_norm(); }

  Quaternion normalized() const { return *this / this->l2_norm(); }

  friend Quaternion operator*(Quaternion lhs, const Quaternion& rhs) {
    const auto w = lhs.w() * rhs.w() - lhs.dot(rhs);
    const auto v = lhs.w() * rhs.v() + rhs.w() * lhs.v() - lhs.cross(rhs);
    lhs._w = w;
    lhs._v = v;
    return lhs;
  }

  static Quaternion AngleAxis(T a, const Vector3<T>& v) {
    if (v.l2_norm_squared() == T{0}) return Quaternion(0, 0, 0, 1);

    Quaternion q;
    const auto rad = a * T{0.5};
    v = v.normalized() * sin(rad);
    q.x() = v.x();
    q.y() = v.y();
    q.z() = v.z();
    q.w() = cos(rad);

    return q.normalized();
  }
};
}  // namespace autd::_utils
