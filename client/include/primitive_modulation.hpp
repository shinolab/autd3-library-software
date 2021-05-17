// File: primitive_modulation.hpp
// Project: include
// Created Date: 14/04/2021
// Author: Shun Suzuki
// -----
// Last Modified: 17/05/2021
// Modified By: Shun Suzuki (suzuki@hapis.k.u-tokyo.ac.jp)
// -----
// Copyright (c) 2021 Hapis Lab. All rights reserved.
//

#pragma once

#include <memory>
#include <vector>

#include "core/configuration.hpp"
#include "core/modulation.hpp"
#include "core/result.hpp"

namespace autd::modulation {

using core::Configuration;
using core::Modulation;
using core::ModulationPtr;

using Static = Modulation;

/**
 * @brief Sine wave modulation
 */
class Sine final : public Modulation {
 public:
  /**
   * @brief Generate function
   * @param[in] freq Frequency of the sine wave
   * @param[in] amp peek to peek amplitude of the wave (Maximum value is 1.0)
   * @param[in] offset offset of the wave
   * @details The sine wave oscillate from offset-amp/2 to offset+amp/2
   */
  static ModulationPtr Create(int freq, double amp = 1.0, double offset = 0.5);
  Error Build(Configuration config) override;
  Sine(const int freq, const double amp, const double offset) : Modulation(), _freq(freq), _amp(amp), _offset(offset) {}

 private:
  int _freq = 0;
  double _amp = 1.0;
  double _offset = 0.5;
};

/**
 * @brief Custom wave modulation
 */
class Custom final : public Modulation {
 public:
  /**
   * @brief Generate function
   * @param[in] buffer data of modulation
   */
  static ModulationPtr Create(const std::vector<uint8_t>& buffer);
  Error Build(Configuration config) override;
  explicit Custom(const std::vector<uint8_t>& buffer) { this->_buffer = buffer; }
};

/**
 * @brief Square wave modulation
 */
class Square final : public Modulation {
 public:
  /**
   * @brief Generate function
   * @param[in] freq Frequency of the square wave
   * @param[in] low low level
   * @param[in] high high level
   */
  static ModulationPtr Create(int freq, uint8_t low = 0, uint8_t high = 0xff);
  Error Build(Configuration config) override;
  Square(const int freq, const uint8_t low, const uint8_t high) : Modulation(), _freq(freq), _low(low), _high(high) {}

 private:
  int _freq = 0;
  uint8_t _low = 0x00;
  uint8_t _high = 0xFF;
};

/**
 * @brief Sawtooth wave modulation
 */
class Saw final : public Modulation {
 public:
  /**
   * @brief Generate function
   * @param[in] freq Frequency of the sawtooth wave
   */
  static ModulationPtr Create(int freq);
  /**
   * @brief Build Modulation
   * @return return Ok(whether succeeded to build), or Err(error msg) if some unrecoverable error occurred
   */
  Error Build(Configuration config) override;
  explicit Saw(const int freq) : Modulation(), _freq(freq) {}

 private:
  int _freq = 0;
};
}  // namespace autd::modulation
