// File: result.hpp
// Project: include
// Created Date: 03/04/2021
// Author: Shun Suzuki
// -----
// Last Modified: 06/04/2021
// Modified By: Shun Suzuki (suzuki@hapis.k.u-tokyo.ac.jp)
// -----
// Copyright (c) 2021 Hapis Lab. All rights reserved.
//

#pragma once

#include <sstream>
#include <stdexcept>
#include <utility>

namespace autd {
template <typename T, typename E>
struct Result {
 private:
  enum class TAG { RESULT_OK, RESULT_ERROR };
  explicit Result(T t) : _t(TAG::RESULT_OK), _ok(std::move(t)), _err() {}
  explicit Result(E e) : _t(TAG::RESULT_ERROR), _ok(), _err(std::move(e)) {}
  TAG _t;
  T _ok;
  E _err;

 public:
  ~Result() = default;
  Result(const Result& obj) { *this = obj; }
  Result& operator=(const Result& obj) {
    _t = obj._t;
    if (_t == TAG::RESULT_OK)
      _ok = obj._ok;
    else if (_t == TAG::RESULT_ERROR)
      _err = obj._err;
    return *this;
  }
  Result(Result&& obj) noexcept { *this = std::move(obj); }
  Result& operator=(Result&& obj) noexcept {
    if (this != &obj) {
      _t = obj._t;
      if (_t == TAG::RESULT_OK)
        _ok = std::move(obj._ok);
      else if (_t == TAG::RESULT_ERROR)
        _err = std::move(obj._err);
    }
    return *this;
  }

  static Result Ok(T ok) { return Result(std::move(ok)); }
  static Result Err(E err) { return Result(std::move(err)); }

  [[nodiscard]] bool is_ok() const { return _t == TAG::RESULT_OK; }
  [[nodiscard]] bool is_err() const { return _t == TAG::RESULT_ERROR; }

  T unwrap() {
    if (_t != TAG::RESULT_OK) {
      std::stringstream ss;
      ss << "cannot unwrap: " << _err;
      throw std::runtime_error(ss.str());
    }
    return std::move(_ok);
  }

  E unwrap_err() {
    if (_t != TAG::RESULT_ERROR) throw std::runtime_error("cannot unwrap_err");
    return std::move(_err);
  }

  [[nodiscard]] T unwrap_or(T v) {
    if (_t != TAG::RESULT_OK) return v;
    return std::move(_ok);
  }
};

template <typename T>
struct OkType {
  explicit OkType(T t) : _t(std::move(t)) {}

  template <typename E>
  operator Result<T, E>() {
    return Result<T, E>::Ok(std::move(_t));
  }

 private:
  T _t;
};

template <typename T>
struct ErrType {
  explicit ErrType(T t) : _t(std::move(t)) {}

  template <typename V>
  operator Result<V, T>() {
    return Result<V, T>::Err(std::move(_t));
  }

 private:
  T _t;
};

template <typename T>
OkType<T> Ok(T t) {
  return OkType<T>(std::move(t));
}

template <typename T>
ErrType<T> Err(T t) {
  return ErrType<T>(std::move(t));
}
}  // namespace autd
