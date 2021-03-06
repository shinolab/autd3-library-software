// File: vector.hpp
// Project: include
// Created Date: 27/02/2020
// Author: Shun Suzuki
// -----
// Last Modified: 06/03/2021
// Modified By: Shun Suzuki (suzuki@hapis.k.u-tokyo.ac.jp)
// -----
// Copyright (c) 2020 Hapis Lab. All rights reserved.
//

#pragma once

#define _USE_MATH_DEFINES
#include <math.h>

#include <cmath>
#include <cstring>
#include <iostream>
#include <memory>

#include "helper.hpp"

namespace autd::_utils {

template <typename T>
struct VectorX {
 public:
  explicit VectorX(const size_t size) : _size(size) { _data = std::make_unique<T[]>(size); }
  VectorX(const VectorX& obj) : VectorX(obj.size()) { std::memcpy(_data.get(), obj.data(), _size * sizeof(T)); }
  VectorX& operator=(const VectorX& obj) {
    std::memcpy(_data.get(), obj.data(), _size * sizeof(T));
    return *this;
  }

  T l2_norm_squared() const { return _Helper::l2_norm_squared<T, VectorX>(*this); }
  T l2_norm() const { return std::sqrt(l2_norm_squared()); }
  T norm() const { return l2_norm(); }

  void normalize() { *this /= l2_norm(); }
  VectorX& normalized() const { return *this / this->l2_norm(); }

  static VectorX Zero(size_t size) {
    VectorX v(size);
    auto* p = v.data();
    for (size_t i = 0; i < v.size(); i++) *p++ = T{0};
    return v;
  }
  static VectorX Ones(size_t size) {
    VectorX v(size);
    for (size_t i = 0; i < size; i++) v._data[i] = T{1};
    return v;
  }

  T dot(const VectorX& rhs) const { return _Helper::dot<T, VectorX, VectorX>(*this, rhs); }

  T& at(size_t i) { return _data[i]; }
  const T& at(size_t i) const { return _data[i]; }

  T& operator()(size_t i) { return _data[i]; }
  const T& operator()(size_t i) const { return _data[i]; }

  T* data() { return _data.get(); }
  const T* data() const { return _data.get(); }

  size_t size() const noexcept { return _size; }
  template <typename Ts>
  friend std::ostream& operator<<(std::ostream&, const VectorX<Ts>&);
  template <typename Ts>
  friend bool operator==(const VectorX<Ts>& lhs, const VectorX<Ts>& rhs);
  template <typename Ts>
  friend bool operator!=(const VectorX<Ts>& lhs, const VectorX<Ts>& rhs);

  VectorX& operator+=(const VectorX& rhs) { return _Helper::add<T, VectorX>(*this, rhs); }
  VectorX& operator-=(const VectorX& rhs) { return _Helper::sub<T, VectorX>(*this, rhs); }
  VectorX& operator*=(T rhs) { return _Helper::mul<T, VectorX>(*this, rhs); }
  VectorX& operator/=(T rhs) { return _Helper::div<T, VectorX>(*this, rhs); }

  VectorX operator-() const { return _Helper::neg<T, VectorX>(*this); }

  friend VectorX operator+(const VectorX& lhs, const VectorX& rhs) { return _Helper::add<T, VectorX>(lhs, rhs); }

  friend VectorX operator-(const VectorX& lhs, const VectorX& rhs) { return _Helper::sub<T, VectorX>(lhs, rhs); }
  friend VectorX operator*(const VectorX& lhs, const T& rhs) { return _Helper::mul<T, VectorX>(lhs, rhs); }
  friend VectorX operator*(const T& lhs, const VectorX& rhs) { return _Helper::mul<T, VectorX>(rhs, lhs); }
  friend VectorX operator/(const VectorX& lhs, const T& rhs) { return _Helper::div<T, VectorX>(lhs, rhs); }

 protected:
  size_t _size;
  std::unique_ptr<T[]> _data;
};

template <typename T>
std::ostream& operator<<(std::ostream& os, const VectorX<T>& obj) {
  return _Helper::vec_show(os, obj);
}
template <typename T>
bool operator==(const VectorX<T>& lhs, const VectorX<T>& rhs) {
  return _Helper::vec_equals(lhs, rhs);
}
template <typename T>
bool operator!=(const VectorX<T>& lhs, const VectorX<T>& rhs) {
  return !(lhs == rhs);
}

template <typename T>
class Vector3 : public VectorX<T> {
 public:
  Vector3() : VectorX<T>(3) {}
  Vector3(T x, T y, T z) : VectorX<T>(3) {
    this->_data[0] = x;
    this->_data[1] = y;
    this->_data[2] = z;
  }

  T& x() noexcept { return this->at(0); }
  T& y() noexcept { return this->at(1); }
  T& z() noexcept { return this->at(2); }
  const T& x() const noexcept { return this->at(0); }
  const T& y() const noexcept { return this->at(1); }
  const T& z() const noexcept { return this->at(2); }

  static Vector3 UnitX() { return Vector3(1, 0, 0); }
  static Vector3 UnitY() { return Vector3(0, 1, 0); }
  static Vector3 UnitZ() { return Vector3(0, 0, 1); }
  static Vector3 Zero() { return Vector3(0, 0, 0); }

  Vector3 normalized() const { return *this / this->l2_norm(); }

  Vector3 cross(const Vector3& rhs) const {
    return Vector3(y() * rhs.z() - z() * rhs.y(), z() * rhs.x() - x() * rhs.z(), x() * rhs.y() - y() * rhs.x());
  }

  T angle(const Vector3& v) const {
    auto cos = this->dot(v) / (this->l2_norm() * v.l2_norm());
    if (cos > 1) {
      return 0.0;
    } else if (cos < -1) {
      return M_PI;
    } else {
      return acos(cos);
    }
  }

  Vector3& operator+=(const Vector3& rhs) { return _Helper::add<T, Vector3>(*this, rhs); }
  Vector3& operator-=(const Vector3& rhs) { return _Helper::sub<T, Vector3>(*this, rhs); }
  Vector3& operator*=(const T& rhs) { return _Helper::mul<T, Vector3>(*this, rhs); }
  Vector3& operator/=(const T& rhs) { return _Helper::div<T, Vector3>(*this, rhs); }

  Vector3 operator-() const { return _Helper::neg<T, Vector3>(*this); }

  friend Vector3 operator+(const Vector3& lhs, const Vector3& rhs) { return _Helper::add<T, Vector3>(lhs, rhs); }

  friend Vector3 operator-(const Vector3& lhs, const Vector3& rhs) { return _Helper::sub<T, Vector3>(lhs, rhs); }
  friend Vector3 operator*(const Vector3& lhs, const T& rhs) { return _Helper::mul<T, Vector3>(lhs, rhs); }
  friend Vector3 operator*(const T& lhs, const Vector3& rhs) { return _Helper::mul<T, Vector3>(rhs, lhs); }
  friend Vector3 operator/(const Vector3& lhs, const T& rhs) { return _Helper::div<T, Vector3>(lhs, rhs); }
};

template <typename T>
class Vector4 : public VectorX<T> {
 public:
  Vector4() : VectorX<T>(4) {}
  Vector4(T x, T y, T z, T w) : VectorX<T>(4) {
    this->_data[0] = x;
    this->_data[1] = y;
    this->_data[2] = z;
    this->_data[3] = w;
  }

  T& x() noexcept { return this->at(0); }
  T& y() noexcept { return this->at(1); }
  T& z() noexcept { return this->at(2); }
  T& w() noexcept { return this->at(3); }
  const T& x() const noexcept { return this->at(0); }
  const T& y() const noexcept { return this->at(1); }
  const T& z() const noexcept { return this->at(2); }
  const T& w() const noexcept { return this->at(3); }

  static Vector4 UnitX() { return Vector4(1, 0, 0, 0); }
  static Vector4 UnitY() { return Vector4(0, 1, 0, 0); }
  static Vector4 UnitZ() { return Vector4(0, 0, 1, 0); }
  static Vector4 UnitW() { return Vector4(0, 0, 0, 1); }
  static Vector4 Zero() { return Vector4(0, 0, 0, 0); }

  Vector4 normalized() const { return *this / this->l2_norm(); }

  Vector4& operator+=(const Vector4& rhs) { return _Helper::add<T, Vector4>(*this, rhs); }
  Vector4& operator-=(const Vector4& rhs) { return _Helper::sub<T, Vector4>(*this, rhs); }
  Vector4& operator*=(const T& rhs) { return _Helper::mul<T, Vector4>(*this, rhs); }
  Vector4& operator/=(const T& rhs) { return _Helper::div<T, Vector4>(*this, rhs); }

  Vector4 operator-() const { return _Helper::neg<T, Vector4>(*this); }

  friend Vector4 operator+(const Vector4& lhs, const Vector4& rhs) { return _Helper::add<T, Vector4>(lhs, rhs); }

  friend Vector4 operator-(const Vector4& lhs, const Vector4& rhs) { return _Helper::sub<T, Vector4>(lhs, rhs); }
  friend Vector4 operator*(const Vector4& lhs, const T& rhs) { return _Helper::mul<T, Vector4>(lhs, rhs); }
  friend Vector4 operator*(const T& lhs, const Vector4& rhs) { return _Helper::mul<T, Vector4>(rhs, lhs); }
  friend Vector4 operator/(const Vector4& lhs, const T& rhs) { return _Helper::div<T, Vector4>(lhs, rhs); }
};

}  // namespace autd::_utils
