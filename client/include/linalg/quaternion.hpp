// File: quaternion.hpp
// Project: include
// Created Date: 27/02/2020
// Author: Shun Suzuki
// -----
// Last Modified: 25/02/2021
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
  Quaternion() = default;
  Quaternion(T w, T x, T y, T z) {
    this->_v = Vector3(x, y, z);
    this->_w = w;
  }
  Quaternion(T w, const Vector3<T>& v) {
    this->_v = v;
    this->_w = w;
  }
  Quaternion(const Quaternion& v) = default;
  Quaternion& operator=(const Quaternion& obj) = default;

  Vector3<T>& v() noexcept { return _v; }
  const Vector3<T>& v() const noexcept { return _v; }
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
  Quaternion conjugate() const {
    Quaternion q = *this;
    Vector3<T> mv = -_v;
    q._v = mv;
    return q;
  }
  Quaternion inverse() const { return conjugate() / l2_norm_squared(); }

  friend Quaternion operator*(Quaternion lhs, const Quaternion& rhs) {
    const auto w = lhs.w() * rhs.w() - lhs.v().dot(rhs.v());
    const auto v = lhs.w() * rhs.v() + rhs.w() * lhs.v() - lhs.v().cross(rhs.v());
    lhs._w = w;
    lhs._v = v;
    return lhs;
  }

  friend Vector3<T> operator*(Quaternion lhs, const Vector3<T>& rhs) {
    const auto p = Quaternion(0, rhs);
    const auto r = lhs * p * lhs.inverse();
    return r.v();
  }

  static Quaternion AngleAxis(T a, const Vector3<T>& v) {
    if (v.l2_norm_squared() == T{0}) return Quaternion(0, 0, 0, 1);

    const auto rad = a * T{0.5};
    const auto vn = v.normalized() * sin(rad);

    Quaternion q(cos(rad), vn);

    return q.normalized();
  }

  template <typename Ts>
  friend inline std::ostream& operator<<(std::ostream&, const Quaternion<Ts>&);
  template <typename Ts>
  friend inline bool operator==(const Quaternion<Ts>& lhs, const Quaternion<Ts>& rhs);
  template <typename Ts>
  friend inline bool operator!=(const Quaternion<Ts>& lhs, const Quaternion<Ts>& rhs);

  Quaternion& operator+=(const Quaternion& rhs) {
    _w += rhs.w();
    _v += rhs.v();
    return *this;
  }
  Quaternion& operator-=(const Quaternion& rhs) {
    _w -= rhs.w();
    _v -= rhs.v();
    return *this;
  }
  Quaternion& operator*=(T rhs) {
    _w *= rhs;
    _v *= rhs;
    return *this;
  }
  Quaternion& operator/=(T rhs) {
    _w /= rhs;
    _v /= rhs;
    return *this;
  }

  friend Quaternion operator+(Quaternion lhs, const Quaternion& rhs) {
    lhs += rhs;
    return lhs;
  }
  friend Quaternion operator-(Quaternion lhs, const Quaternion& rhs) {
    lhs -= rhs;
    return lhs;
  }
  friend Quaternion operator*(Quaternion lhs, const T& rhs) {
    lhs *= rhs;
    return lhs;
  }
  friend Quaternion operator*(const T& lhs, Quaternion rhs) {
    rhs *= lhs;
    return rhs;
  }
  friend Quaternion operator/(Quaternion lhs, const T& rhs) {
    lhs /= rhs;
    return lhs;
  }
};

template <typename T>
inline std::ostream& operator<<(std::ostream& os, const Quaternion<T>& obj) {
  os << obj.w() << " + " << obj.x() << "i + " << obj.y() << "j + " << obj.z() << "k";
  return os;
}
template <typename T>
inline bool operator==(const Quaternion<T>& lhs, const Quaternion<T>& rhs) {
  return lhs.w() == rhs.w() && lhs.v() == rhs.v();
}
template <typename T>
inline bool operator!=(const Quaternion<T>& lhs, const Quaternion<T>& rhs) {
  return !(lhs == rhs);
}

}  // namespace autd::_utils
