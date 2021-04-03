// File: matrix.hpp
// Project: include
// Created Date: 27/02/2020
// Author: Shun Suzuki
// -----
// Last Modified: 04/04/2021
// Modified By: Shun Suzuki (suzuki@hapis.k.u-tokyo.ac.jp)
// -----
// Copyright (c) 2020 Hapis Lab. All rights reserved.
//

#pragma once

#include <algorithm>
#include <cstring>
#include <memory>

#include "helper.hpp"
#include "quaternion.hpp"
#include "vector.hpp"

namespace autd::_utils {

template <typename T>
struct MatrixX {
  MatrixX(const size_t row, const size_t col) : _num_row(row), _num_col(col) { _data = std::make_unique<T[]>(row * col); }
  ~MatrixX() = default;
  MatrixX(const MatrixX& obj) {
    _num_row = obj._num_row;
    _num_col = obj._num_col;
    std::memcpy(_data.get(), obj.data(), size() * sizeof(T));
  }
  MatrixX& operator=(const MatrixX& obj) {
    std::memcpy(_data.get(), obj.data(), size() * sizeof(T));
    return *this;
  }
  MatrixX(MatrixX&& obj) = default;
  MatrixX& operator=(MatrixX&& obj) = default;

  static MatrixX Zero(size_t row, size_t col) {
    MatrixX v(row, col);
    auto* p = v.data();
    for (size_t i = 0; i < v.size(); i++) *p++ = T{0};
    return v;
  }

  static MatrixX Identity(const size_t row, const size_t col) {
    MatrixX v = MatrixX::Zero(row, col);
    for (size_t i = 0; i < std::min(row, col); i++) v(i, i) = T{1};
    return v;
  }

  T& at(const size_t row, const size_t col) { return _data[col * _num_row + row]; }
  [[nodiscard]] const T& at(const size_t row, const size_t col) const { return _data[col * _num_row + row]; }

  T& operator()(const size_t row, const size_t col) { return _data[col * _num_row + row]; }
  const T& operator()(const size_t row, const size_t col) const { return _data[col * _num_row + row]; }

  T* data() { return _data.get(); }
  [[nodiscard]] const T* data() const { return _data.get(); }

  [[nodiscard]] size_t rows() const noexcept { return _num_row; }
  [[nodiscard]] size_t cols() const noexcept { return _num_col; }
  [[nodiscard]] size_t size() const noexcept { return _num_row * _num_col; }

  [[nodiscard]] VectorX<T> col(const size_t idx) const noexcept {
    VectorX<T> v(_num_row);
    for (size_t i = 0; i < _num_row; i++) {
      v(i) = at(i, idx);
    }
    return v;
  }

  [[nodiscard]] VectorX<T> row(const size_t idx) const noexcept {
    VectorX<T> v(_num_col);
    std::memcpy(v.data(), _data[idx * _num_col], _num_col * sizeof(T));
    return v;
  }

  template <typename Ts>
  friend std::ostream& operator<<(std::ostream&, const MatrixX<Ts>&);
  template <typename Ts>
  friend bool operator==(const MatrixX<Ts>& lhs, const MatrixX<Ts>& rhs);
  template <typename Ts>
  friend bool operator!=(const MatrixX<Ts>& lhs, const MatrixX<Ts>& rhs);

  MatrixX& operator+=(const MatrixX& rhs) { return _Helper::add<T, MatrixX>(*this, rhs); }
  MatrixX& operator-=(const MatrixX& rhs) { return _Helper::sub<T, MatrixX>(*this, rhs); }
  MatrixX& operator*=(T rhs) { return _Helper::mul<T, MatrixX>(*this, rhs); }
  MatrixX& operator/=(T rhs) { return _Helper::div<T, MatrixX>(*this, rhs); }

  MatrixX operator-() const { return _Helper::neg<T, MatrixX>(*this); }

  friend MatrixX operator+(const MatrixX& lhs, const MatrixX& rhs) { return _Helper::add<T, MatrixX>(lhs, rhs); }
  friend MatrixX operator-(const MatrixX& lhs, const MatrixX& rhs) { return _Helper::sub<T, MatrixX>(lhs, rhs); }
  friend MatrixX operator*(const MatrixX& lhs, const T& rhs) { return _Helper::mul<T, MatrixX>(lhs, rhs); }
  friend MatrixX operator*(const T& lhs, const MatrixX& rhs) { return _Helper::mul<T, MatrixX>(rhs, lhs); }
  friend MatrixX operator/(const MatrixX& lhs, const T& rhs) { return _Helper::div<T, MatrixX>(lhs, rhs); }

  friend VectorX<T> operator*(const MatrixX& lhs, const VectorX<T>& rhs) { return _Helper::mat_vec_mul<T, MatrixX, VectorX<T>>(lhs, rhs); }

 private:
  size_t _num_row;
  size_t _num_col;
  std::unique_ptr<T[]> _data;
};

template <typename T>
std::ostream& operator<<(std::ostream& os, const MatrixX<T>& obj) {
  return _Helper::mat_show(os, obj);
}
template <typename T>
bool operator==(const MatrixX<T>& lhs, const MatrixX<T>& rhs) {
  return _Helper::mat_equals(lhs, rhs);
}
template <typename T>
bool operator!=(const MatrixX<T>& lhs, const MatrixX<T>& rhs) {
  return !(lhs == rhs);
}

template <typename T>
class Matrix4x4 : public MatrixX<T> {
 public:
  Matrix4x4() : MatrixX<T>(4, 4) {}

  Matrix4x4& operator+=(const Matrix4x4& rhs) { return _Helper::add<T, Matrix4x4>(*this, rhs); }
  Matrix4x4& operator-=(const Matrix4x4& rhs) { return _Helper::sub<T, Matrix4x4>(*this, rhs); }
  Matrix4x4& operator*=(const T& rhs) { return _Helper::mul<T, Matrix4x4>(*this, rhs); }
  Matrix4x4& operator/=(const T& rhs) { return _Helper::div<T, Matrix4x4>(*this, rhs); }

  Matrix4x4 operator-() const { return _Helper::neg<T, Matrix4x4>(*this); }

  friend Matrix4x4 operator+(const Matrix4x4& lhs, const Matrix4x4& rhs) { return _Helper::add<T, Matrix4x4>(lhs, rhs); }

  friend Matrix4x4 operator-(const Matrix4x4& lhs, const Matrix4x4& rhs) { return _Helper::sub<T, Matrix4x4>(lhs, rhs); }
  friend Matrix4x4 operator*(const Matrix4x4& lhs, const T& rhs) { return _Helper::mul<T, Matrix4x4>(lhs, rhs); }
  friend Matrix4x4 operator*(const T& lhs, Matrix4x4 rhs) { return _Helper::mul<T, Matrix4x4>(rhs, lhs); }
  friend Matrix4x4 operator/(const Matrix4x4& lhs, const T& rhs) { return _Helper::div<T, Matrix4x4>(rhs, lhs); }

  static Matrix4x4 Translation(const Vector3<T>& v, const Quaternion<T>& q) {
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
};

}  // namespace autd::_utils
