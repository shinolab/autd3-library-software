// File: result.hpp
// Project: core
// Created Date: 01/05/2021
// Author: Shun Suzuki
// -----
// Last Modified: 20/05/2021
// Modified By: Shun Suzuki (suzuki@hapis.k.u-tokyo.ac.jp)
// -----
// Copyright (c) 2021 Hapis Lab. All rights reserved.
//

#pragma once

#include <optional>
#include <sstream>
#include <string>
#include <utility>
#include <variant>

namespace autd {
/**
 * \brief Result is a type that contain either success ok(T) or failure err(E).
 * \tparam T type of success data
 * \tparam E type of failure data
 */
template <class T, class E>
class Result : public std::variant<T, E> {
  explicit Result(T t) : std::variant<T, E>(std::forward<T>(t)) {}
  explicit Result(E e) : std::variant<T, E>(std::forward<E>(e)) {}

 public:
  /**
   * \brief Returns true if the result is ok.
   */
  [[nodiscard]] bool is_ok() const { return std::holds_alternative<T>(*this); }
  /**
   * \brief Returns true if the result is err.
   */
  [[nodiscard]] bool is_err() const { return std::holds_alternative<E>(*this); }

  static Result ok(T ok) { return Result(std::forward<T>(ok)); }
  static Result err(E err) { return Result(std::forward<E>(err)); }

  /**
   * \brief Returns the contained ok value. if the contained data is err, this function throw runtime error.
   */
  T unwrap() {
    if (this->is_err()) {
      std::stringstream ss;
      ss << "cannot unwrap: " << std::get<E>(*this);
      throw std::runtime_error(ss.str());
    }
    return std::forward<T>(std::get<T>(*this));
  }

  /**
   * \brief Returns the contained err value. if the contained data is ok, this function throw runtime error.
   */
  E unwrap_err() {
    if (this->is_ok()) throw std::runtime_error("cannot unwrap_err");
    return std::forward<E>(std::get<E>(*this));
  }
};

using Error = Result<bool, std::string>;

/**
 * \brief Type just used for implicit conversion to Result.
 */
template <typename T>
struct OkType {
  explicit OkType(T t) : _t(std::forward<T>(t)) {}

  template <typename E>
  operator Result<T, E>() {
    return Result<T, E>::ok(std::forward<T>(_t));
  }

 private:
  T _t;
};

/**
 * \brief Type just used for implicit conversion to Result.
 */
template <typename T>
struct ErrType {
  explicit ErrType(T t) : _t(std::forward<T>(t)) {}

  template <typename V>
  operator Result<V, T>() {
    return Result<V, T>::err(std::forward<T>(_t));
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
