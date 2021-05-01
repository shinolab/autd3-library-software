// File: result.hpp
// Project: include
// Created Date: 03/04/2021
// Author: Shun Suzuki
// -----
// Last Modified: 01/05/2021
// Modified By: Shun Suzuki (suzuki@hapis.k.u-tokyo.ac.jp)
// -----
// Copyright (c) 2021 Hapis Lab. All rights reserved.
//

#pragma once

#include <sstream>
#include <utility>
#include <variant>

namespace autd {
template <class T, class E>
class Result : public std::variant<T, E> {
 private:
  explicit Result(T t) : std::variant<T, E>(std::forward<T>(t)) {}
  explicit Result(E e) : std::variant<T, E>(std::forward<E>(e)) {}

 public:
  [[nodiscard]] bool is_ok() const { return std::holds_alternative<T>(*this); }
  [[nodiscard]] bool is_err() const { return std::holds_alternative<E>(*this); }

  static Result Ok(T ok) { return Result(std::forward<T>(ok)); }
  static Result Err(E err) { return Result(std::forward<E>(err)); }

  T unwrap() {
    if (this->is_err()) {
      std::stringstream ss;
      ss << "cannot unwrap: " << std::get<E>(*this);
      throw std::runtime_error(ss.str());
    }
    return std::forward<T>(std::get<T>(*this));
  }

  E unwrap_err() {
    if (this->is_ok()) throw std::runtime_error("cannot unwrap_err");
    return std::forward<E>(std::get<E>(*this));
  }

  [[nodiscard]] T unwrap_or(T v) {
    if (this->is_err()) return std::forward<T>(v);
    return std::get<T>(*this);
  }
};

template <typename T>
struct OkType {
  explicit OkType(T t) : _t(std::forward<T>(t)) {}

  template <typename E>
  operator Result<T, E>() {
    return Result<T, E>::Ok(std::forward<T>(_t));
  }

 private:
  T _t;
};

template <typename T>
struct ErrType {
  explicit ErrType(T t) : _t(std::forward<T>(t)) {}

  template <typename V>
  operator Result<V, T>() {
    return Result<V, T>::Err(std::forward<T>(_t));
  }

 private:
  T _t;
};

template <typename T>
OkType<T> Ok(T t) {
  return OkType<T>(std::forward<T>(t));
}

template <typename T>
ErrType<T> Err(T t) {
  return ErrType<T>(std::forward<T>(t));
}
}  // namespace autd
