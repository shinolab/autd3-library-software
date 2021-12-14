// File: type_hints.hpp
// Project: core
// Created Date: 14/12/2021
// Author: Shun Suzuki
// -----
// Last Modified: 14/12/2021
// Modified By: Shun Suzuki (suzuki@hapis.k.u-tokyo.ac.jp)
// -----
// Copyright (c) 2021 Hapis Lab. All rights reserved.
//

#pragma once

#include "autd3/core/gain.hpp"
#include "autd3/core/interface.hpp"

namespace autd::core {

template <typename T, typename B>
struct is_raw_pointer_of : std::is_convertible<T, B*> {};
template <typename T, typename B>
struct is_smart_pointer_of : std::disjunction<std::is_convertible<T, std::unique_ptr<B>>, std::is_convertible<T, std::shared_ptr<B>>> {};
template <typename T, typename B>
struct is_pointer_of : std::disjunction<is_raw_pointer_of<T, B>, is_smart_pointer_of<T, B>> {};

template <typename T>
struct is_body_ref : std::is_base_of<IDatagramBody, std::remove_reference_t<T>> {};
template <typename T>
inline constexpr bool is_body_ref_v = is_body_ref<T>::value;
template <typename T>
struct is_body_ptr : is_pointer_of<T, IDatagramBody> {};
template <typename T>
inline constexpr bool is_body_ptr_v = is_body_ptr<T>::value;
template <typename T>
struct is_body : std::disjunction<is_body_ref<T>, is_body_ptr<T>> {};
template <typename T>
inline constexpr bool is_body_v = is_body<T>::value;

template <typename T>
struct is_header_ref : std::is_base_of<IDatagramHeader, std::remove_reference_t<T>> {};
template <typename T>
inline constexpr bool is_header_ref_v = is_header_ref<T>::value;
template <typename T>
struct is_header_ptr : is_pointer_of<T, IDatagramHeader> {};
template <typename T>
inline constexpr bool is_header_ptr_v = is_header_ptr<T>::value;
template <typename T>
struct is_header : std::disjunction<is_header_ref<T>, is_header_ptr<T>> {};
template <typename T>
inline constexpr bool is_header_v = is_header<T>::value;

template <typename T>
struct is_gain_ref : std::is_base_of<Gain, std::remove_reference_t<T>> {};
template <typename T>
inline constexpr bool is_gain_ref_v = is_gain_ref<T>::value;
template <typename T>
struct is_gain_ptr : is_pointer_of<T, Gain> {};
template <typename T>
inline constexpr bool is_gain_ptr_v = is_gain_ptr<T>::value;
template <typename T>
struct is_gain : std::disjunction<is_gain_ref<T>, is_gain_ptr<T>> {};
template <typename T>
inline constexpr bool is_gain_v = is_gain<T>::value;

template <class T>
std::enable_if_t<is_header_v<T>, IDatagramHeader&> to_header(T&& header) {
  if constexpr (is_header_ref_v<T>)
    return header;
  else
    return *header;
}

template <class T>
std::enable_if_t<is_body_v<T>, IDatagramBody&> to_body(T&& body) {
  if constexpr (is_body_ref_v<T>)
    return body;
  else
    return *body;
}

template <class T>
std::enable_if_t<is_gain_v<T>, Gain&> to_gain(T&& gain) {
  if constexpr (is_gain_ref_v<T>)
    return gain;
  else
    return *gain;
}

}  // namespace autd::core
