// File: primitive.hpp
// Project: modulation
// Created Date: 20/02/2021
// Author: Shun Suzuki
// -----
// Last Modified: 20/02/2021
// Modified By: Shun Suzuki (suzuki@hapis.k.u-tokyo.ac.jp)
// -----
// Copyright (c) 2021 Hapis Lab. All rights reserved.
//

#pragma once

#include "modulation.hpp"

namespace autd::modulation {
/**
 * @brief Sine wave modulation
 */
class SineModulation final : public Modulation {
 public:
  /**
   * @brief Generate function
   * @param[in] freq Frequency of the sine wave
   * @param[in] amp peek to peek amplitude of the wave (Maximum value is 1.0)
   * @param[in] offset offset of the wave
   * @details The sine wave oscillate from offset-amp/2 to offset+amp/2
   */
  static ModulationPtr Create(int freq, Float amp = 1.0, Float offset = 0.5);
  void Build(Configuration config) override;
  SineModulation(const int freq, const Float amp, const Float offset) : Modulation(), _freq(freq), _amp(amp), _offset(offset) {}

 private:
  int _freq = 0;
  Float _amp = 1.0;
  Float _offset = 0.5;
};

/**
 * @brief Square wave modulation
 */
class SquareModulation final : public Modulation {
 public:
  /**
   * @brief Generate function
   * @param[in] freq Frequency of the square wave
   * @param[in] low low level
   * @param[in] high high level
   */
  static ModulationPtr Create(int freq, uint8_t low = 0, uint8_t high = 0xff);
  void Build(Configuration config) override;
  SquareModulation(const int freq, const uint8_t low, const uint8_t high) : Modulation(), _freq(freq), _low(low), _high(high) {}

 private:
  int _freq = 0;
  uint8_t _low = 0x00;
  uint8_t _high = 0xFF;
};

/**
 * @brief Sawtooth wave modulation
 */
class SawModulation final : public Modulation {
 public:
  /**
   * @brief Generate function
   * @param[in] freq Frequency of the sawtooth wave
   */
  static ModulationPtr Create(int freq);
  void Build(Configuration config) override;
  explicit SawModulation(const int freq) : Modulation(), _freq(freq) {}

 private:
  int _freq = 0;
};
}  // namespace autd::modulation
