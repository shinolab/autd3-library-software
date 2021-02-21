// File: primitive.cpp
// Project: primitive
// Created Date: 20/02/2021
// Author: Shun Suzuki
// -----
// Last Modified: 20/02/2021
// Modified By: Shun Suzuki (suzuki@hapis.k.u-tokyo.ac.jp)
// -----
// Copyright (c) 2021 Hapis Lab. All rights reserved.
//

#include "modulation/primitive.hpp"

#include <algorithm>
#include <cstring>
#include <numeric>

namespace autd::modulation {
#pragma region SineModulation
ModulationPtr SineModulation::Create(const int freq, const Float amp, const Float offset) {
  ModulationPtr mod = std::make_shared<SineModulation>(freq, amp, offset);
  return mod;
}

void SineModulation::Build(const Configuration config) {
  const auto sf = static_cast<int32_t>(config.mod_sampling_freq());
  const auto mod_buf_size = static_cast<int32_t>(config.mod_buf_size());

  const auto freq = std::clamp(this->_freq, 1, sf / 2);

  const auto d = std::gcd(sf, freq);

  const size_t n = mod_buf_size / d / (mod_buf_size / sf);
  const size_t rep = freq / d;

  this->buffer.resize(n, 0);

  const auto offset = static_cast<double>(this->_offset);
  const auto amp = static_cast<double>(this->_amp);
  for (size_t i = 0; i < n; i++) {
    auto tamp = fmod(static_cast<Float>(2 * rep * i) / static_cast<Float>(n), 2.0);
    tamp = tamp > 1.0 ? 2.0 - tamp : tamp;
    tamp = std::clamp(offset + (tamp - 0.5) * amp, 0.0, 1.0);
    this->buffer.at(i) = static_cast<uint8_t>(tamp * 255.0);
  }
}
#pragma endregion

#pragma region SquareModulation
ModulationPtr SquareModulation::Create(int freq, uint8_t low, uint8_t high) {
  ModulationPtr mod = std::make_shared<SquareModulation>(freq, low, high);
  return mod;
}

void SquareModulation::Build(const Configuration config) {
  const auto sf = static_cast<int32_t>(config.mod_sampling_freq());
  const auto mod_buf_size = static_cast<int32_t>(config.mod_buf_size());

  const auto freq = std::clamp(this->_freq, 1, sf / 2);

  const auto d = std::gcd(sf, freq);

  const size_t n = mod_buf_size / d / (mod_buf_size / sf);

  this->buffer.resize(n, this->_high);
  std::memset(&this->buffer[0], this->_low, n / 2);
}
#pragma endregion

#pragma region SawModulation
ModulationPtr SawModulation::Create(const int freq) {
  ModulationPtr mod = std::make_shared<SawModulation>(freq);
  return mod;
}

void SawModulation::Build(const Configuration config) {
  const auto sf = static_cast<int32_t>(config.mod_sampling_freq());
  const auto mod_buf_size = static_cast<int32_t>(config.mod_buf_size());

  const auto freq = std::clamp(this->_freq, 1, sf / 2);

  const auto d = std::gcd(sf, freq);

  const size_t n = mod_buf_size / d / (mod_buf_size / sf);
  const auto rep = freq / d;

  this->buffer.resize(n, 0);

  for (size_t i = 0; i < n; i++) {
    const auto tamp = fmod(static_cast<double>(rep * i) / static_cast<double>(n), 1.0);
    this->buffer.at(i) = static_cast<uint8_t>(asin(tamp) / M_PI * 510.0);
  }
}
#pragma endregion
}  // namespace autd::modulation