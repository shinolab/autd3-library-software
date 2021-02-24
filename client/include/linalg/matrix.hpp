// File: matrix4x4.hpp
// Project: linalg
// Created Date: 22/02/2021
// Author: Shun Suzuki
// -----
// Last Modified: 24/02/2021
// Modified By: Shun Suzuki (suzuki@hapis.k.u-tokyo.ac.jp)
// -----
// Copyright (c) 2021 Hapis Lab. All rights reserved.
//

#pragma once

#define _USE_MATH_DEFINES
#include <math.h>

#include <cmath>
#include <cstring>
#include <iostream>
#include <memory>

namespace autd::_utils {
template <typename T>
struct Matrix {
  Matrix() = default;
  Matrix(size_t num_row, size_t num_col) : _num_row(num_row), _num_col(num_col) { _data = std::make_unique<T[]>(num_row * num_col); }
  Matrix(const Matrix& obj) : Matrix(obj._num_row, obj._num_col) { std::memcpy(_data.get(), obj.data(), _num_row * _num_col * sizeof(T)); }
  Matrix& operator=(const Matrix& obj) { std::memcpy(_data.get(), obj.data(), _num_row * _num_col * sizeof(T)); };

  T& operator()(size_t row, size_t col) { return _data[col * _num_row + row]; }
  const T& operator()(size_t row, size_t col) const { return _data[col * _num_row + row]; }

  T& at(size_t row, size_t col) { return _data[col * _num_row + row]; }
  const T& at(size_t row, size_t col) const { return _data[col * _num_row + row]; }

  T* data() { return _data.get(); }
  const T* data() const { return _data.get(); }

  T frobenius_norm_squared() const {
    T n = 0;
    for (auto i = 0; i < _num_row * _num_col; i++) n += _data[i] * _data[i];
    return n;
  }
  T frobenius_norm() const { return std::sqrt(frobenius_norm_squared()); }

  static Matrix Zero() noexcept {
    Matrix v;
    std::memset(v._data, 0, _num_row * _num_col * sizeof(T));
    return v;
  }

  template <typename Ts>
  friend inline std::ostream& operator<<(std::ostream&, const Matrix<Ts>&);
  template <typename Ts>
  friend inline bool operator==(const Matrix<Ts>& lhs, const Matrix<Ts>& rhs);
  template <typename Ts>
  friend inline bool operator!=(const Matrix<Ts>& lhs, const Matrix<Ts>& rhs);

  Matrix& operator+=(const Matrix& rhs) {
    for (auto i = 0; i < _num_row * _num_col; i++) this->_data[i] += rhs._data[i];
    return *this;
  }
  Matrix& operator-=(const Matrix& rhs) {
    for (auto i = 0; i < _num_row * _num_col; i++) this->_data[i] -= rhs._data[i];
    return *this;
  }
  Matrix& operator*=(const T& rhs) {
    for (auto i = 0; i < _num_row * _num_col; i++) this->_data[i] *= rhs;
    return *this;
  }
  Matrix& operator/=(const T& rhs) {
    for (auto i = 0; i < _num_row * _num_col; i++) this->_data[i] /= rhs;
    return *this;
  }

  friend Matrix operator+(Matrix lhs, const Matrix& rhs) {
    lhs += rhs;
    return lhs;
  }
  friend Matrix operator-(Matrix lhs, const Matrix& rhs) {
    lhs -= rhs;
    return lhs;
  }
  friend Matrix operator*(Matrix lhs, const T& rhs) {
    lhs *= rhs;
    return lhs;
  }
  friend Matrix operator*(const T& lhs, Matrix rhs) {
    rhs *= lhs;
    return rhs;
  }
  friend Matrix operator/(Matrix lhs, const T& rhs) {
    lhs /= rhs;
    return lhs;
  }

  VectorX<T> col(size_t idx) {
    VectorX<T> v = VectorX<T>(_num_row);
    std::memcpy(v.data(), &_data[idx * _num_row], _num_row * sizeof(T));
    return v;
  }

  const size_t num_rows() { return _num_row; }
  const size_t num_cols() { return _num_col; }
  const size_t size() { return _num_col * _num_row; }

 private:
  size_t _num_row;
  size_t _num_col;
  std::unique_ptr<T[]> _data;
};

template <typename T>
inline std::ostream& operator<<(std::ostream& os, const Matrix<T>& obj) {
  os << "Matrix" << obj.num_rows() << "x" << obj.num_cols() << ":";
  for (auto row = 0; row < obj.num_rows(); row++) {
    os << "\n\t";
    for (auto col = 0; col < obj.num_cols(); col++) os << obj(row, col) << ", ";
  }
  return os;
}
template <typename T>
inline bool operator==(const Matrix<T>& lhs, const Matrix<T>& rhs) {
  if (lhs.num_rows() != rhs.num_rows()) return false;
  if (lhs.num_cols() != rhs.num_cols()) return false;
  bool r = true;
  for (auto col = 0; col < obj.num_cols(); col++)
    for (auto row = 0; row < obj.num_rows(); row++) r = r && (lhs(row, col) == rhs(row, col));
  return r;
}
template <typename T>
inline bool operator!=(const Matrix<T>& lhs, const Matrix<T>& rhs) {
  return !(lhs == rhs);
}

template <typename T>
class Matrix4x4 : public Matrix<T> {
 public:
  Matrix4x4() : Matrix(4, 4) {}
};

}  // namespace autd::_utils
