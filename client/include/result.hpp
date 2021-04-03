// File: result.hpp
// Project: include
// Created Date: 03/04/2021
// Author: Shun Suzuki
// -----
// Last Modified: 03/04/2021
// Modified By: Shun Suzuki (suzuki@hapis.k.u-tokyo.ac.jp)
// -----
// Copyright (c) 2021 Hapis Lab. All rights reserved.
//

#pragma once

#include <stdexcept>

template <typename T, typename E>
struct Result {
 private:
  enum class tag { RESULT_OK, RESULT_ERROR };
  explicit Result(const tag t) : _t(t), _ok() {}
  tag _t;
  union {
    T _ok;
    E _err;
  };

 public:
  ~Result() {
    if (_t == tag::RESULT_OK) {
      _ok.~T();
    } else {
      _err.~E();
    }
  }
  Result(const Result& obj) : _t(obj._t) {
    if (_t == tag::RESULT_OK) {
      _ok = obj._ok;
    } else {
      _err = obj._err;
    }
  }
  Result& operator=(const Result& obj) { return *this; }
  Result(Result&& obj) = default;
  Result& operator=(Result&& obj) = default;

  static Result Ok(const T& ok) {
    Result result(tag::RESULT_OK);
    result._ok = ok;
    return result;
  }

  static Result Err(const E& err) {
    Result result(tag::RESULT_ERROR);
    result._err = err;
    return result;
  }

  [[nodiscard]] bool is_ok() const { return _t == tag::RESULT_OK; }
  [[nodiscard]] bool is_err() const { return _t == tag::RESULT_ERROR; }

  [[nodiscard]] T const& unwrap() const {
    if (_t != tag::RESULT_OK) throw std::runtime_error("cannot unwrap");

    return _ok;
  }

  [[nodiscard]] E const& unwrap_err() const {
    if (_t != tag::RESULT_ERROR) throw std::runtime_error("cannot unwrap_err");

    return _err;
  }
};

template <typename T>
struct _Ok {
  explicit _Ok(T t) : _t(t) {}

  template <typename E>
  operator Result<T, E>() const {
    return Result<T, E>::Ok(_t);
  }

 private:
  T _t;
};

template <typename T>
struct _Err {
  explicit _Err(T t) : _t(t) {}

  template <typename V>
  operator Result<V, T>() const {
    return Result<V, T>::Err(_t);
  }

 private:
  T _t;
};

template <typename T>
_Ok<T> Ok(T t) {
  return _Ok<T>(t);
}

template <typename T>
_Err<T> Err(T t) {
  return _Err<T>(t);
}
