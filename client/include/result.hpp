// File: result.hpp
// Project: include
// Created Date: 03/04/2021
// Author: Shun Suzuki
// -----
// Last Modified: 04/04/2021
// Modified By: Shun Suzuki (suzuki@hapis.k.u-tokyo.ac.jp)
// -----
// Copyright (c) 2021 Hapis Lab. All rights reserved.
//

#pragma once

#include <sstream>
#include <stdexcept>
#include <utility>

template <typename T, typename E>
struct Result {
 private:
  enum class tag { RESULT_OK, RESULT_ERROR };
  explicit Result(T t) : _t(tag::RESULT_OK), _ok(std::move(t)) {}
  explicit Result(E e) : _t(tag::RESULT_ERROR), _err(std::move(e)) {}
  tag _t;
  union {
    T _ok;
    E _err;
  };

 public:
  ~Result() { _t == tag::RESULT_OK ? _ok.~T() : _err.~E(); }
  Result(const Result& obj) : _t(obj._t) {
    if (_t == tag::RESULT_OK)
      _ok = obj._ok;
    else
      _err = obj._err;
  }
  Result& operator=(const Result& obj) {
    _t = obj._t;
    if (_t == tag::RESULT_OK)
      _ok = obj._ok;
    else
      _err = obj._err;
    return *this;
  }
  Result(Result&& obj) noexcept(false) {
    _t = obj._t;
    if (_t == tag::RESULT_OK)
      _ok = std::move(obj._ok);
    else
      _err = std::move(obj._err);
  }
  Result& operator=(Result&& obj) noexcept(false) {
    _t == tag::RESULT_OK ? _ok.~T() : _err.~E();
    _t = obj._t;
    if (_t == tag::RESULT_OK)
      _ok = std::move(obj._ok);
    else
      _err = std::move(obj._err);
    return *this;
  }

  static Result Ok(T ok) { return Result(std::move(ok)); }
  static Result Err(E err) { return Result(std::move(err)); }

  [[nodiscard]] bool is_ok() const { return _t == tag::RESULT_OK; }
  [[nodiscard]] bool is_err() const { return _t == tag::RESULT_ERROR; }

  T unwrap() {
    if (_t != tag::RESULT_OK) {
      std::stringstream ss;
      ss << "cannot unwrap: " << _err;
      throw std::runtime_error(ss.str());
    }
    return std::move(_ok);
  }

  E unwrap_err() {
    if (_t != tag::RESULT_ERROR) throw std::runtime_error("cannot unwrap_err");
    return std::move(_err);
  }

  [[nodiscard]] T unwrap_or(T v) {
    if (_t != tag::RESULT_OK) return v;
    return std::move(_ok);
  }
};

template <typename T>
struct _Ok {
  explicit _Ok(T t) : _t(std::move(t)) {}

  template <typename E>
  operator Result<T, E>() {
    return Result<T, E>::Ok(std::move(_t));
  }

 private:
  T _t;
};

template <typename T>
struct _Err {
  explicit _Err(T t) : _t(std::move(t)) {}

  template <typename V>
  operator Result<V, T>() {
    return Result<V, T>::Err(std::move(_t));
  }

 private:
  T _t;
};

template <typename T>
_Ok<T> Ok(T t) {
  return _Ok<T>(std::move(t));
}

template <typename T>
_Err<T> Err(T t) {
  return _Err<T>(std::move(t));
}
