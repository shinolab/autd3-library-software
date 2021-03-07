// File: helper.hpp
// Project: eigen-linalg
// Created Date: 25/02/2021
// Author: Shun Suzuki
// -----
// Last Modified: 08/03/2021
// Modified By: Shun Suzuki (suzuki@hapis.k.u-tokyo.ac.jp)
// -----
// Copyright (c) 2021 Hapis Lab. All rights reserved.
//

#pragma once

#include <ostream>

namespace autd::_utils {

class _Helper {
 public:
  template <typename T, typename V>
  static V& add(V& dst, const V& src) {
    T* dp = dst.data();
    const T* sp = src.data();
    for (size_t i = 0; i < dst.size(); i++) *dp++ += *sp++;
    return dst;
  }
  template <typename T, typename V>
  static V add(const V& lhs, const V& rhs) {
    V dst(lhs);
    T* dp = dst.data();
    const T* rp = rhs.data();
    for (size_t i = 0; i < dst.size(); i++) *dp++ += *rp++;
    return dst;
  }
  template <typename T, typename V>
  static V neg(const V& src) {
    V dst(src);
    T* dp = dst.data();
    for (size_t i = 0; i < dst.size(); i++) {
      *dp = -*dp;
      ++dp;
    }
    return dst;
  }

  template <typename T, typename V>
  static V& sub(V& dst, const V& src) {
    T* dp = dst.data();
    const T* sp = src.data();
    for (size_t i = 0; i < dst.size(); i++) *dp++ -= *sp++;
    return dst;
  }

  template <typename T, typename V>
  static V sub(const V& lhs, const V& rhs) {
    V dst(lhs);
    T* dp = dst.data();
    const T* rp = rhs.data();
    for (size_t i = 0; i < dst.size(); i++) *dp++ -= *rp++;
    return dst;
  }

  template <typename T, typename V>
  static V& mul(V& dst, const T& src) {
    T* dp = dst.data();
    for (size_t i = 0; i < dst.size(); i++) *dp++ *= src;
    return dst;
  }

  template <typename T, typename V>
  static V mul(const V& lhs, const T& rhs) {
    V dst(lhs);
    T* dp = dst.data();
    for (size_t i = 0; i < dst.size(); i++) *dp++ *= rhs;
    return dst;
  }

  template <typename T, typename V>
  static V& div(V& dst, const T& src) {
    T* dp = dst.data();
    for (size_t i = 0; i < dst.size(); i++) *dp++ /= src;
    return dst;
  }

  template <typename T, typename V>
  static V div(const V& lhs, const T& rhs) {
    V dst(lhs);
    T* dp = dst.data();
    for (size_t i = 0; i < dst.size(); i++) *dp++ /= rhs;
    return dst;
  }

  template <typename V>
  static bool vec_equals(const V& lhs, const V& rhs) {
    if (lhs.size() != rhs.size()) return false;
    auto r = true;
    const auto* lp = lhs.data();
    const auto* rp = rhs.data();
    for (size_t i = 0; i < lhs.size(); i++) r = r && (*lp++ == *rp++);
    return r;
  }

  template <typename V>
  static bool mat_equals(const V& lhs, const V& rhs) {
    if (lhs.cols() != rhs.cols()) return false;
    if (lhs.rows() != rhs.rows()) return false;
    auto r = true;
    const auto* lp = lhs.data();
    const auto* rp = rhs.data();
    for (size_t i = 0; i < lhs.size(); i++) r = r && (*lp++ == *rp++);
    return r;
  }

  template <typename V>
  static std::ostream& mat_show(std::ostream& os, const V& obj) {
    os << "Matrix" << obj.rows() << "x" << obj.cols() << ":";
    for (size_t row = 0; row < obj.rows(); row++) {
      os << "\n\t";
      for (size_t col = 0; col < obj.cols(); col++) os << obj(row, col) << ", ";
    }
    return os;
  }

  template <typename V>
  static std::ostream& vec_show(std::ostream& os, const V& obj) {
    os << "Vector" << obj.size() << ":";
    for (size_t i = 0; i < obj.size(); i++) os << "\n\t" << obj(i);
    return os;
  }

  template <typename T, typename V1, typename V2>
  static T dot(const V1& lhs, const V2& rhs) {
    T d = 0;
    const T* lp = lhs.data();
    const T* rp = rhs.data();
    for (size_t i = 0; i < lhs.size(); i++) d += *(lp++) * *(rp++);
    return d;
  }

  template <typename T, typename V>
  static T l2_norm_squared(const V& v) {
    T n = 0;
    const T* lp = v.data();
    for (size_t i = 0; i < v.size(); i++) {
      n += *lp * *lp;
      ++lp;
    }
    return n;
  }

  template <typename T, typename M, typename V>
  static V mat_vec_mul(const M& m, const V& v) {
    V res(v.size());
    for (size_t row = 0; row < m.rows(); row++) {
      T n = 0;
      for (size_t col = 0; col < m.cols(); col++) {
        n += m(row, col) * v(col);
      }
      res(row) = n;
    }

    return res;
  }
};

}  // namespace autd::_utils
