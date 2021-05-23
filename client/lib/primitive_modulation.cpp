// File: primitive_modulation.cpp
// Project: lib
// Created Date: 14/04/2021
// Author: Shun Suzuki
// -----
// Last Modified: 20/05/2021
// Modified By: Shun Suzuki (suzuki@hapis.k.u-tokyo.ac.jp)
// -----
// Copyright (c) 2021 Hapis Lab. All rights reserved.
//

#include "primitive_modulation.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <numeric>

#include "core/configuration.hpp"
#include "core/result.hpp"

namespace autd::modulation {

using core::Configuration;

ModulationPtr Sine::create(const int freq, const double amp, const double offset) { return std::make_shared<Sine>(freq, amp, offset); }

Error Sine::build(const Configuration config) {
  const auto sf = static_cast<int32_t>(config.mod_sampling_freq());
  const auto mod_buf_size = static_cast<int32_t>(config.mod_buf_size());

  const auto freq = std::clamp(this->_freq, 1, sf / 2);

  const auto d = std::gcd(sf, freq);

  const size_t n = mod_buf_size / d / (mod_buf_size / sf);
  const size_t rep = freq / d;

  this->_buffer.resize(n, 0);

  for (size_t i = 0; i < n; i++) {
    auto tamp = std::fmod(static_cast<double>(2 * rep * i) / static_cast<double>(n), double{2});
    tamp = tamp > 1 ? 2 - tamp : tamp;
    tamp = std::clamp(this->_offset + (tamp - double{0.5}) * this->_amp, double{0}, double{1});
    this->_buffer.at(i) = static_cast<uint8_t>(tamp * 255);
  }
  return Ok(true);
}

ModulationPtr Square::create(int freq, uint8_t low, uint8_t high) { return std::make_shared<Square>(freq, low, high); }

Error Square::build(const Configuration config) {
  const auto sf = static_cast<int32_t>(config.mod_sampling_freq());
  const auto mod_buf_size = static_cast<int32_t>(config.mod_buf_size());

  const auto freq = std::clamp(this->_freq, 1, sf / 2);

  const auto d = std::gcd(sf, freq);

  const size_t n = mod_buf_size / d / (mod_buf_size / sf);

  this->_buffer.resize(n, this->_high);
  std::memset(&this->_buffer[0], this->_low, n / 2);
  return Ok(true);
}

ModulationPtr Custom::create(const std::vector<uint8_t>& buffer) { return std::make_shared<Custom>(buffer); }

Error Custom::build(const Configuration config) {
  (void)config;
  return Ok(true);
}

ModulationPtr Saw::create(const int freq) { return std::make_shared<Saw>(freq); }

Error Saw::build(const Configuration config) {
  const auto sf = static_cast<int32_t>(config.mod_sampling_freq());
  const auto mod_buf_size = static_cast<int32_t>(config.mod_buf_size());

  const auto freq = std::clamp(this->_freq, 1, sf / 2);

  const auto d = std::gcd(sf, freq);

  const size_t n = mod_buf_size / d / (mod_buf_size / sf);
  const auto rep = freq / d;

  this->_buffer.resize(n, 0);

  for (size_t i = 0; i < n; i++) {
    const auto tamp = std::fmod(static_cast<double>(rep * i) / static_cast<double>(n), double{1});
    this->_buffer.at(i) = static_cast<uint8_t>(std::asin(tamp) / M_PI * double{510});
  }
  return Ok(true);
}
}  // namespace autd::modulation
