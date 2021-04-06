// File: vector.hpp
// Project: include
// Created Date: 27/02/2020
// Author: Shun Suzuki
// -----
// Last Modified: 06/04/2021
// Modified By: Shun Suzuki (suzuki@hapis.k.u-tokyo.ac.jp)
// -----
// Copyright (c) 2020 Hapis Lab. All rights reserved.
//

#pragma once

#include <cmath>
#include <cstring>
#include <memory>

#include "consts.hpp"
#include "helper.hpp"

namespace autd::utils {

template <typename T>
struct VectorX {
  explicit VectorX(const size_t size) : _size(size) { _data = std::make_unique<T[]>(size); }
  ~VectorX() = default;
  VectorX(const VectorX& obj) : VectorX(obj.size()) { *this = obj; }
  VectorX& operator=(const VectorX& obj) {
    std::memcpy(_data.get(), obj.data(), _size * sizeof(T));
    return *this;
  }
  VectorX(const VectorX&& obj) noexcept { *this = std::move(obj); }
  VectorX& operator=(VectorX&& obj) noexcept {
    if (this != &obj) {
      _size = obj._size;
      _data = std::move(obj._data);
    }
    return *this;
  }

  [[nodiscard]] T l2_norm_squared() const { return LinalgHelper::l2_norm_squared<T, VectorX>(*this); }
  [[nodiscard]] T l2_norm() const { return std::sqrt(l2_norm_squared()); }
  [[nodiscard]] T norm() const { return l2_norm(); }

  void normalize() { *this /= l2_norm(); }
  [[nodiscard]] VectorX& normalized() const { return *this / this->l2_norm(); }

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

  [[nodiscard]] T dot(const VectorX& rhs) const { return LinalgHelper::dot<T, VectorX, VectorX>(*this, rhs); }

  T& at(size_t i) { return _data[i]; }
  [[nodiscard]] const T& at(size_t i) const { return _data[i]; }

  T& operator()(size_t i) { return _data[i]; }
  const T& operator()(size_t i) const { return _data[i]; }

  T* data() { return _data.get(); }
  [[nodiscard]] const T* data() const { return _data.get(); }

  [[nodiscard]] size_t size() const noexcept { return _size; }
  template <typename Ts>
  friend std::ostream& operator<<(std::ostream&, const VectorX<Ts>&);
  template <typename Ts>
  friend bool operator==(const VectorX<Ts>& lhs, const VectorX<Ts>& rhs);
  template <typename Ts>
  friend bool operator!=(const VectorX<Ts>& lhs, const VectorX<Ts>& rhs);

  VectorX& operator+=(const VectorX& rhs) { return LinalgHelper::add<T, VectorX>(this, rhs); }
  VectorX& operator-=(const VectorX& rhs) { return LinalgHelper::sub<T, VectorX>(this, rhs); }
  VectorX& operator*=(T rhs) { return LinalgHelper::mul<T, VectorX>(this, rhs); }
  VectorX& operator/=(T rhs) { return LinalgHelper::div<T, VectorX>(this, rhs); }

  VectorX operator-() const { return LinalgHelper::neg<T, VectorX>(*this); }

  friend VectorX operator+(const VectorX& lhs, const VectorX& rhs) { return LinalgHelper::add<T, VectorX>(lhs, rhs); }
  friend VectorX operator-(const VectorX& lhs, const VectorX& rhs) { return LinalgHelper::sub<T, VectorX>(lhs, rhs); }
  friend VectorX operator*(const VectorX& lhs, const T& rhs) { return LinalgHelper::mul<T, VectorX>(lhs, rhs); }
  friend VectorX operator*(const T& lhs, const VectorX& rhs) { return LinalgHelper::mul<T, VectorX>(rhs, lhs); }
  friend VectorX operator/(const VectorX& lhs, const T& rhs) { return LinalgHelper::div<T, VectorX>(lhs, rhs); }

 protected:
  size_t _size;
  std::unique_ptr<T[]> _data;
};

template <typename T>
std::ostream& operator<<(std::ostream& os, const VectorX<T>& obj) {
  return LinalgHelper::vec_show(os, obj);
}
template <typename T>
bool operator==(const VectorX<T>& lhs, const VectorX<T>& rhs) {
  return LinalgHelper::vec_equals(lhs, rhs);
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
    }

    if (cos < -1) {
      return PI;
    }

    return acos(cos);
  }

  Vector3& operator+=(const Vector3& rhs) { return LinalgHelper::add<T, Vector3>(this, rhs); }
  Vector3& operator-=(const Vector3& rhs) { return LinalgHelper::sub<T, Vector3>(this, rhs); }
  Vector3& operator*=(const T& rhs) { return LinalgHelper::mul<T, Vector3>(this, rhs); }
  Vector3& operator/=(const T& rhs) { return LinalgHelper::div<T, Vector3>(this, rhs); }

  Vector3 operator-() const { return LinalgHelper::neg<T, Vector3>(*this); }

  friend Vector3 operator+(const Vector3& lhs, const Vector3& rhs) { return LinalgHelper::add<T, Vector3>(lhs, rhs); }

  friend Vector3 operator-(const Vector3& lhs, const Vector3& rhs) { return LinalgHelper::sub<T, Vector3>(lhs, rhs); }
  friend Vector3 operator*(const Vector3& lhs, const T& rhs) { return LinalgHelper::mul<T, Vector3>(lhs, rhs); }
  friend Vector3 operator*(const T& lhs, const Vector3& rhs) { return LinalgHelper::mul<T, Vector3>(rhs, lhs); }
  friend Vector3 operator/(const Vector3& lhs, const T& rhs) { return LinalgHelper::div<T, Vector3>(lhs, rhs); }
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

  Vector4& operator+=(const Vector4& rhs) { return LinalgHelper::add<T, Vector4>(this, rhs); }
  Vector4& operator-=(const Vector4& rhs) { return LinalgHelper::sub<T, Vector4>(this, rhs); }
  Vector4& operator*=(const T& rhs) { return LinalgHelper::mul<T, Vector4>(this, rhs); }
  Vector4& operator/=(const T& rhs) { return LinalgHelper::div<T, Vector4>(this, rhs); }

  Vector4 operator-() const { return LinalgHelper::neg<T, Vector4>(*this); }

  friend Vector4 operator+(const Vector4& lhs, const Vector4& rhs) { return LinalgHelper::add<T, Vector4>(lhs, rhs); }

  friend Vector4 operator-(const Vector4& lhs, const Vector4& rhs) { return LinalgHelper::sub<T, Vector4>(lhs, rhs); }
  friend Vector4 operator*(const Vector4& lhs, const T& rhs) { return LinalgHelper::mul<T, Vector4>(lhs, rhs); }
  friend Vector4 operator*(const T& lhs, const Vector4& rhs) { return LinalgHelper::mul<T, Vector4>(rhs, lhs); }
  friend Vector4 operator/(const Vector4& lhs, const T& rhs) { return LinalgHelper::div<T, Vector4>(lhs, rhs); }
};

}  // namespace autd::utils
