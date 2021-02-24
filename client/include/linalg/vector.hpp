// File: vector.hpp
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

#define _USE_MATH_DEFINES
#include <math.h>

#include <cmath>
#include <cstring>
#include <iostream>
#include <memory>

namespace autd::_utils {
class VectorHelper {
 public:
  template <typename T, typename V1, typename V2>
  static T dot(const V1& lhs, const V2& rhs) {
    T d = 0;
    const T* lp = lhs.get();
    const T* rp = rhs.get();
    for (auto i = 0; i < lhs.size(); i++) d += *(lp++) * *(rp++);
    return d;
  }

  template <typename T, typename V>
  static T l2_norm_squared(V& v) {
    T n = 0;
    auto lp = lhs.get();
    for (auto i = 0; i < v.size(); i++) n += *lp * *(lp++);
    return n;
  }

  template <typename T, typename V>
  static V& add(V& dst, const V& src) {
    T* dp = dst.data();
    const T* sp = src.data();
    for (auto i = 0; i < dst.size(); i++) *dp++ += *sp++;
    return dst;
  }

  template <typename T, typename V>
  static V& sub(V& dst, const V& src) {
    T* dp = dst.data();
    const T* sp = src.data();
    for (auto i = 0; i < dst.size(); i++) *dp++ -= *sp++;
    return dst;
  }

  template <typename T, typename V>
  static V& mul(V& dst, const T& src) {
    T* dp = dst.data();
    for (auto i = 0; i < dst.size(); i++) *dp++ *= src;
    return dst;
  }

  template <typename T, typename V>
  static V div(V& dst, const T& src) {
    T* dp = dst.data();
    for (auto i = 0; i < dst.size(); i++) *dp++ /= src;
    return dst;
  }

  template <typename V>
  static bool equals(const V& lhs, const V& rhs) {
    if (lhs.size() != rhs.size()) return false;
    bool r = true;
    for (auto i = 0; i < lhs.size(); i++) r = r && (lhs(i) == rhs(i));
    return r;
  }

  template <typename V>
  static std::ostream& show(std::ostream& os, const V& obj) {
    os << "Vector" << obj.size() << ":";
    for (auto i = 0; i < obj.size(); i++) os << "\n\t" << obj(i);
    return os;
  }
};

template <typename T>
struct VectorX {
 public:
  VectorX() = default;
  VectorX(size_t size) : _size(size) { _data = std::make_unique<T[]>(size); }
  VectorX(const VectorX& obj) : VectorX(obj.size()) { std::memcpy(_data.get(), obj.data(), _size * sizeof(T)); }
  VectorX& operator=(const VectorX& obj) { std::memcpy(_data.get(), obj.data(), _size * sizeof(T)); };

  T l2_norm_squared() const { return VectorHelper::l2_norm_squared(this); }
  T l2_norm() const { return std::sqrt(l2_norm_squared()); }
  T norm() const { return l2_norm(); }

  void normalize() { *this /= l2_norm(); }
  VectorX& normalized() const { return *this / this->l2_norm(); }

  static VectorX Zero() {
    VectorX v;
    std::memset(v._data, 0, _size * sizeof(T));
    return v;
  }

  T dot(const VectorX& rhs) const { return VectorHelper::dot<T, VectorX, VectorX>(*this, rhs); }

  T& at(size_t i) { return _data[i]; }
  const T& at(size_t i) const { return _data[i]; }

  T& operator()(size_t i) { return _data[i]; }
  const T& operator()(size_t i) const { return _data[i]; }

  T* data() { return _data.get(); }
  const T* data() const { return _data.get(); }

  const size_t size() const noexcept { return _size; }

  template <typename Ts>
  friend inline std::ostream& operator<<(std::ostream&, const VectorX<Ts>&);
  template <typename Ts>
  friend inline bool operator==(const VectorX<Ts>& lhs, const VectorX<Ts>& rhs);
  template <typename Ts>
  friend inline bool operator!=(const VectorX<Ts>& lhs, const VectorX<Ts>& rhs);

  VectorX& operator+=(const VectorX& rhs) { return VectorHelper::add(this, rhs); }
  VectorX& operator-=(const VectorX& rhs) { return VectorHelper::sub(this, rhs); }
  VectorX& operator*=(T rhs) { return VectorHelper::mul(this, rhs); }
  VectorX& operator/=(T rhs) { return VectorHelper::div(this, rhs); }

  friend VectorX operator+(VectorX lhs, const VectorX& rhs) {
    lhs += rhs;
    return lhs;
  }
  friend VectorX operator-(VectorX lhs, const VectorX& rhs) {
    lhs -= rhs;
    return lhs;
  }
  friend VectorX operator*(VectorX lhs, const T& rhs) {
    lhs *= rhs;
    return lhs;
  }
  friend VectorX operator*(const T& lhs, VectorX rhs) {
    rhs *= lhs;
    return rhs;
  }
  friend VectorX operator/(VectorX lhs, const T& rhs) {
    lhs /= rhs;
    return lhs;
  }

 private:
  size_t _size;
  std::unique_ptr<T[]> _data;
};

template <typename T>
inline std::ostream& operator<<(std::ostream& os, const VectorX<T>& obj) {
  return VectorHelper::show(os, obj);
}
template <typename T>
inline bool operator==(const VectorX<T>& lhs, const VectorX<T>& rhs) {
  return VectorHelper::equals(lhs, rhs);
}
template <typename T>
inline bool operator!=(const VectorX<T>& lhs, const VectorX<T>& rhs) {
  return !(lhs == rhs);
}

template <typename T>
class Vector3 : public VectorX<T> {
 public:
  Vector3(T x, T y, T z) : VectorX(3) {
    data()[0] = x;
    data()[1] = y;
    data()[2] = z;
  }

  T& x() noexcept { return at(0); }
  T& y() noexcept { return at(1); }
  T& z() noexcept { return at(2); }
  const T& x() const noexcept { return at(0); }
  const T& y() const noexcept { return at(1); }
  const T& z() const noexcept { return at(2); }

  static Vector3 UnitX() { return Vector3(1, 0, 0); }
  static Vector3 UnitY() { return Vector3(0, 1, 0); }
  static Vector3 UnitZ() { return Vector3(0, 0, 1); }
  static Vector3 Zero() { return Vector3(0, 0, 0); }

  Vector3& normalized() const { return *this / this->l2_norm(); }

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

  template <typename Ts>
  friend inline std::ostream& operator<<(std::ostream&, const Vector3<Ts>&);
  template <typename Ts>
  friend inline bool operator==(const Vector3<Ts>& lhs, const Vector3<Ts>& rhs);
  template <typename Ts>
  friend inline bool operator!=(const Vector3<Ts>& lhs, const Vector3<Ts>& rhs);

  Vector3& operator+=(const Vector3& rhs) { return VectorHelper::add<T, Vector3>(*this, rhs); }
  Vector3& operator-=(const Vector3& rhs) { return VectorHelper::sub<T, Vector3>(*this, rhs); }
  Vector3& operator*=(const T& rhs) { return VectorHelper::mul<T, Vector3>(*this, rhs); }
  Vector3& operator/=(const T& rhs) { return VectorHelper::div<T, Vector3>(*this, rhs); }

  friend Vector3 operator+(Vector3 lhs, const Vector3& rhs) {
    lhs += rhs;
    return lhs;
  }
  friend Vector3 operator-(Vector3 lhs, const Vector3& rhs) {
    lhs -= rhs;
    return lhs;
  }
  friend Vector3 operator*(Vector3 lhs, const T& rhs) {
    lhs *= rhs;
    return lhs;
  }
  friend Vector3 operator*(const T& lhs, Vector3 rhs) {
    rhs *= lhs;
    return rhs;
  }
  friend Vector3 operator/(Vector3 lhs, const T& rhs) {
    lhs /= rhs;
    return lhs;
  }
};

template <typename T>
inline std::ostream& operator<<(std::ostream& os, const Vector3<T>& obj) {
  return VectorHelper::show(os, obj);
}
template <typename T>
inline bool operator==(const Vector3<T>& lhs, const Vector3<T>& rhs) {
  return VectorHelper::equals(lhs, rhs);
}
template <typename T>
inline bool operator!=(const Vector3<T>& lhs, const Vector3<T>& rhs) {
  return !(lhs == rhs);
}
}  // namespace autd::_utils
