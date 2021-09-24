// File: primitive.cpp
// Project: modulation
// Created Date: 14/04/2021
// Author: Shun Suzuki
// -----
// Last Modified: 24/09/2021
// Modified By: Shun Suzuki (suzuki@hapis.k.u-tokyo.ac.jp)
// -----
// Copyright (c) 2021 Hapis Lab. All rights reserved.
//

#include "autd3/modulation/primitive.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <numeric>

#include "autd3/core/utils.hpp"

namespace autd::modulation {

ModulationPtr Sine::create(const int freq, const double amp, const double offset) { return std::make_shared<Sine>(freq, amp, offset); }

void Sine::calc() {
  const auto f_s = static_cast<int32_t>(sampling_freq());

  const auto f = std::clamp(this->_freq, 1, f_s / 2);

  const auto k = std::gcd(f_s, f);

  const size_t n = f_s / k;
  const size_t d = f / k;

  this->_buffer.resize(n, 0);
  for (size_t i = 0; i < n; i++) {
    const auto amp = this->_amp / 2.0 * std::sin(2.0 * M_PI * static_cast<double>(d * i) / static_cast<double>(n)) + this->_offset;
    this->_buffer[i] = core::Utilities::to_duty(amp);
  }
}

ModulationPtr SinePressure::create(const int freq, const double amp, const double offset) {
  return std::make_shared<SinePressure>(freq, amp, offset);
}

void SinePressure::calc() {
  const auto f_s = static_cast<int32_t>(sampling_freq());

  const auto f = std::clamp(this->_freq, 1, f_s / 2);

  const auto k = std::gcd(f_s, f);

  const size_t n = f_s / k;
  const size_t d = f / k;

  this->_buffer.resize(n, 0);
  for (size_t i = 0; i < n; i++) {
    const auto amp = std::sqrt(this->_amp / 2.0 * std::sin(2.0 * M_PI * static_cast<double>(d * i) / static_cast<double>(n)) + this->_offset);
    this->_buffer[i] = core::Utilities::to_duty(amp);
  }
}

ModulationPtr SineLegacy::create(const double freq, const double amp, const double offset) { return std::make_shared<SineLegacy>(freq, amp, offset); }

void SineLegacy::calc() {
  const auto f_s = sampling_freq();
  const auto f =
      std::clamp(this->_freq, static_cast<double>(core::MOD_SAMPLING_FREQ_BASE) / static_cast<double>(core::MOD_SAMPLING_FREQ_DIV_MAX), f_s / 2.0);

  const auto t = static_cast<size_t>(std::round(f_s / f));
  this->_buffer.resize(t, 0);
  for (size_t i = 0; i < t; i++) {
    double tamp = 255.0 * _offset + 127.5 * _amp * std::cos(2.0 * M_PI * static_cast<double>(i) / static_cast<double>(t));
    this->_buffer[i] = static_cast<uint8_t>(std::clamp(tamp, 0.0, 255.0));
  }
}

ModulationPtr Square::create(const int freq, const uint8_t low, const uint8_t high) { return std::make_shared<Square>(freq, low, high); }
ModulationPtr Square::create(const int freq, const double low, const double high) {
  return create(freq, core::Utilities::to_duty(low), core::Utilities::to_duty(high));
}

void Square::calc() {
  const auto sf = static_cast<int32_t>(sampling_freq());
  const auto freq = std::clamp(this->_freq, 1, sf / 2);
  const auto d = std::gcd(sf, freq);
  const size_t n = sf / d;

  std::fill(this->_buffer.begin(), this->_buffer.end(), this->_high);
  this->_buffer.resize(n, this->_high);
  std::memset(&this->_buffer[0], this->_low, n / 2);
}

ModulationPtr Custom::create(const std::vector<uint8_t>& buffer) { return std::make_shared<Custom>(buffer); }

void Custom::calc() {}
}  // namespace autd::modulation
