// File: vector3.hpp
// Project: include
// Created Date: 27/02/2020
// Author: Shun Suzuki
// -----
// Last Modified: 25/12/2020
// Modified By: Shun Suzuki (suzuki@hapis.k.u-tokyo.ac.jp)
// -----
// Copyright (c) 2020 Hapis Lab. All rights reserved.
//

#pragma once

#define _USE_MATH_DEFINES  // NOLINT
#include <math.h>

#include <iostream>

namespace autd {
namespace utils {
/**
 * @brief Simple three-dimensional vector class
 */
class Vector3 {
 public:
  Vector3() noexcept = default;
  ~Vector3() noexcept = default;
  Vector3(const double x, const double y, const double z) noexcept : _x(x), _y(y), _z(z) {}
  Vector3(const Vector3& v) noexcept = default;
  Vector3& operator=(const Vector3& obj) = default;
  Vector3(Vector3&& obj) = default;
  Vector3& operator=(Vector3&& obj) = default;

  [[nodiscard]] double x() const noexcept { return _x; }
  [[nodiscard]] double y() const noexcept { return _y; }
  [[nodiscard]] double z() const noexcept { return _z; }
  [[nodiscard]] double l2_norm_squared() const { return _x * _x + _y * _y + _z * _z; }
  [[nodiscard]] double l2_norm() const { return std::sqrt(l2_norm_squared()); }

  static Vector3 unit_x() noexcept { return Vector3(1, 0, 0); }
  static Vector3 unit_y() noexcept { return Vector3(0, 1, 0); }
  static Vector3 unit_z() noexcept { return Vector3(0, 0, 1); }
  static Vector3 zero() noexcept { return Vector3(0, 0, 0); }

  [[nodiscard]] Vector3 normalized() const { return *this / this->l2_norm(); }

  [[nodiscard]] double dot(const Vector3& rhs) const { return _x * rhs._x + _y * rhs._y + _z * rhs._z; }
  [[nodiscard]] Vector3 cross(const Vector3& rhs) const {
    return Vector3(_y * rhs._z - _z * rhs._y, _z * rhs._x - _x * rhs._z, _x * rhs._y - _y * rhs._x);
  }

  [[nodiscard]] double angle(const Vector3& v) const {
    const auto cos = this->dot(v) / (this->l2_norm() * v.l2_norm());
    if (cos > 1) {
      return 0.0;
    }

    if (cos < -1) {
      return M_PI;
    }

    return acos(cos);
  }

  friend inline std::ostream& operator<<(std::ostream&, const Vector3&);

  Vector3& operator+=(const Vector3& rhs) {
    this->_x += rhs._x;
    this->_y += rhs._y;
    this->_z += rhs._z;
    return *this;
  }
  Vector3& operator-=(const Vector3& rhs) {
    this->_x -= rhs._x;
    this->_y -= rhs._y;
    this->_z -= rhs._z;
    return *this;
  }
  Vector3& operator*=(const double& rhs) {
    this->_x *= rhs;
    this->_y *= rhs;
    this->_z *= rhs;
    return *this;
  }
  Vector3& operator/=(const double& rhs) {
    this->_x /= rhs;
    this->_y /= rhs;
    this->_z /= rhs;
    return *this;
  }

  friend Vector3 operator+(Vector3 lhs, const Vector3& rhs) {
    lhs += rhs;
    return lhs;
  }
  friend Vector3 operator-(Vector3 lhs, const Vector3& rhs) {
    lhs -= rhs;
    return lhs;
  }
  friend Vector3 operator*(Vector3 lhs, const double& rhs) {
    lhs *= rhs;
    return lhs;
  }
  friend Vector3 operator*(const double& lhs, Vector3 rhs) {
    rhs *= lhs;
    return rhs;
  }
  friend Vector3 operator/(Vector3 lhs, const double& rhs) {
    lhs /= rhs;
    return lhs;
  }

 private:
  double _x;
  double _y;
  double _z;
};

inline std::ostream& operator<<(std::ostream& os, const Vector3& obj) {
  os << "Vector3 {x: " << obj._x << ", y: " << obj._y << ", z: " << obj._z << "}";
  return os;
}
}  // namespace utils
}  // namespace autd
