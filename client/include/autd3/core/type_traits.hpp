// File: type_traits.hpp
// Project: core
// Created Date: 14/12/2021
// Author: Shun Suzuki
// -----
// Last Modified: 16/12/2021
// Modified By: Shun Suzuki (suzuki@hapis.k.u-tokyo.ac.jp)
// -----
// Copyright (c) 2021 Hapis Lab. All rights reserved.
//

#pragma once

#include <memory>
#include <type_traits>

#include "autd3/core/gain.hpp"
#include "autd3/core/interface.hpp"

namespace autd::core::type_traits {

/**
 * @brief Checks if a type is raw pointer
 */
template <typename T, typename B>
struct is_raw_pointer_of : std::is_convertible<T, B*> {};
/**
 * @brief Checks if a type is smart pointer
 */
template <typename T, typename B>
struct is_smart_pointer_of : std::disjunction<std::is_convertible<T, std::unique_ptr<B>>, std::is_convertible<T, std::shared_ptr<B>>> {};
/**
 * @brief Checks if a type is pointer
 */
template <typename T, typename B>
struct is_pointer_of : std::disjunction<is_raw_pointer_of<T, B>, is_smart_pointer_of<T, B>> {};

/**
 * @brief Checks if a type is reference of IDatagramBody
 */
template <typename T>
struct is_body_ref : std::is_base_of<datagram::IDatagramBody, std::remove_reference_t<T>> {};
template <typename T>
inline constexpr bool is_body_ref_v = is_body_ref<T>::value;
/**
 * @brief Checks if a type is pointer of IDatagramBody
 */
template <typename T>
struct is_body_ptr : is_pointer_of<T, datagram::IDatagramBody> {};
template <typename T>
inline constexpr bool is_body_ptr_v = is_body_ptr<T>::value;
/**
 * @brief Checks if a type is IDatagramBody
 */
template <typename T>
struct is_body : std::disjunction<is_body_ref<T>, is_body_ptr<T>> {};
template <typename T>
inline constexpr bool is_body_v = is_body<T>::value;

/**
 * @brief Checks if a type is reference of IDatagramHeader
 */
template <typename T>
struct is_header_ref : std::is_base_of<datagram::IDatagramHeader, std::remove_reference_t<T>> {};
template <typename T>
inline constexpr bool is_header_ref_v = is_header_ref<T>::value;
/**
 * @brief Checks if a type is pointer of IDatagramHeader
 */
template <typename T>
struct is_header_ptr : is_pointer_of<T, datagram::IDatagramHeader> {};
template <typename T>
inline constexpr bool is_header_ptr_v = is_header_ptr<T>::value;
/**
 * @brief Checks if a type is IDatagramHeader
 */
template <typename T>
struct is_header : std::disjunction<is_header_ref<T>, is_header_ptr<T>> {};
template <typename T>
inline constexpr bool is_header_v = is_header<T>::value;

/**
 * @brief Checks if a type is reference of Gain
 */
template <typename T>
struct is_gain_ref : std::is_base_of<Gain, std::remove_reference_t<T>> {};
template <typename T>
inline constexpr bool is_gain_ref_v = is_gain_ref<T>::value;
/**
 * @brief Checks if a type is pointer of Gain
 */
template <typename T>
struct is_gain_ptr : is_pointer_of<T, Gain> {};
template <typename T>
inline constexpr bool is_gain_ptr_v = is_gain_ptr<T>::value;
/**
 * @brief Checks if a type is Gain
 */
template <typename T>
struct is_gain : std::disjunction<is_gain_ref<T>, is_gain_ptr<T>> {};
template <typename T>
inline constexpr bool is_gain_v = is_gain<T>::value;

template <class T>
std::enable_if_t<is_header_v<T>, datagram::IDatagramHeader&> to_header(T&& header) {
  if constexpr (is_header_ref_v<T>)
    return header;
  else
    return *header;
}

template <class T>
std::enable_if_t<is_body_v<T>, datagram::IDatagramBody&> to_body(T&& body) {
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

}  // namespace autd::core::type_traits
