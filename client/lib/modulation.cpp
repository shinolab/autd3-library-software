// File: modulation.cpp
// Project: lib
// Created Date: 11/06/2016
// Author: Seki Inoue
// -----
// Last Modified: 06/03/2021
// Modified By: Shun Suzuki (suzuki@hapis.k.u-tokyo.ac.jp)
// -----
// Copyright (c) 2016-2020 Hapis Lab. All rights reserved.
//

#include "modulation.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <numeric>

#include "configuration.hpp"

namespace autd::modulation {
Modulation::Modulation() noexcept { this->_sent = 0; }

ModulationPtr Modulation::Create(const uint8_t amp) {
  auto mod = std::make_shared<Modulation>();
  mod->buffer.resize(1, amp);
  return mod;
}

void Modulation::Build(Configuration config) {}

size_t& Modulation::sent() { return _sent; }

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

  for (size_t i = 0; i < n; i++) {
    auto tamp = std::fmod(static_cast<Float>(2 * rep * i) / static_cast<Float>(n), Float{2});
    tamp = tamp > Float{1} ? Float{2} - tamp : tamp;
    tamp = std::clamp(this->_offset + (tamp - Float{0.5}) * this->_amp, Float{0}, Float{1});
    this->buffer.at(i) = static_cast<uint8_t>(tamp * 255);
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
