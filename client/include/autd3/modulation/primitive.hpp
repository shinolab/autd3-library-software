// File: primitive_modulation.hpp
// Project: include
// Created Date: 14/04/2021
// Author: Shun Suzuki
// -----
// Last Modified: 22/11/2021
// Modified By: Shun Suzuki (suzuki@hapis.k.u-tokyo.ac.jp)
// -----
// Copyright (c) 2021 Hapis Lab. All rights reserved.
//

#pragma once

#include <memory>
#include <vector>

#include "autd3/core/modulation.hpp"

namespace autd::modulation {

using core::Modulation;
using core::ModulationPtr;

using Static = Modulation;

/**
 * @brief Sine wave modulation in ultrasound amplitude
 */
class Sine final : public Modulation {
 public:
  /**
   * @brief Generate function
   * @param[in] freq Frequency of the sine wave
   * @param[in] amp peek to peek ultrasound amplitude (Maximum value is 1.0)
   * @param[in] offset offset of ultrasound amplitude
   * @details Ultrasound amplitude oscillate from offset-amp/2 to offset+amp/2.
   * If the value exceeds the range of [0, 1], the value will be clamped in the [0, 1].
   */
  static ModulationPtr create(int freq, double amp = 1.0, double offset = 0.5);
  void calc() override;
  Sine(const int freq, const double amp, const double offset) : Modulation(), _freq(freq), _amp(amp), _offset(offset) {}

 private:
  int _freq;
  double _amp;
  double _offset;
};

/**
 * @brief Sine wave modulation in radiation pressure
 */
class SinePressure final : public Modulation {
 public:
  /**
   * @brief Generate function
   * @param[in] freq Frequency of the sine wave
   * @param[in] amp peek to peek amplitude of radiation pressure (Maximum value is 1.0)
   * @param[in] offset offset of radiation pressure
   * @details Radiation pressure oscillate from offset-amp/2 to offset+amp/2
   * If the value exceeds the range of [0, 1], the value will be clamped in the [0, 1].
   */
  static ModulationPtr create(int freq, double amp = 1.0, double offset = 0.5);
  void calc() override;
  SinePressure(const int freq, const double amp, const double offset) : Modulation(), _freq(freq), _amp(amp), _offset(offset) {}

 private:
  int _freq;
  double _amp;
  double _offset;
};

/**
 * @brief Sine wave modulation in ultrasound amplitude (Legacy)
 */
class SineLegacy final : public Modulation {
 public:
  /**
   * @brief Generate function
   * @param[in] freq Frequency of the sine wave
   * @param[in] amp peek to peek ultrasound amplitude (Maximum value is 1.0)
   * @param[in] offset offset of ultrasound amplitude
   * @details Ultrasound amplitude oscillate from offset-amp/2 to offset+amp/2.
   * If the value exceeds the range of [0, 1], the value will be clamped in the [0, 1].
   */
  static ModulationPtr create(double freq, double amp = 1.0, double offset = 0.5);
  void calc() override;
  SineLegacy(const double freq, const double amp, const double offset) : Modulation(), _freq(freq), _amp(amp), _offset(offset) {}

 private:
  double _freq;
  double _amp;
  double _offset;
};

/**
 * @brief Square wave modulation
 */
class Square final : public Modulation {
 public:
  /**
   * @brief Generate function
   * @param[in] freq Frequency of the square wave
   * @param[in] low low level in duty
   * @param[in] high high level in duty
   * @param[in] duty duty ratio of square wave
   */
  static ModulationPtr create(int freq, uint8_t low = 0, uint8_t high = 0xff, double duty = 0.5);

  /**
   * @brief Generate function
   * @param[in] freq Frequency of the square wave
   * @param[in] low low level in relative amplitude (0 to 1)
   * @param[in] high high level in relative amplitude (0 to 1)
   * @param[in] duty duty ratio of square wave
   */
  static ModulationPtr create(int freq, double low, double high, double duty);

  void calc() override;
  Square(const int freq, const uint8_t low, const uint8_t high, const double duty) : Modulation(), _freq(freq), _low(low), _high(high), _duty(duty) {}

 private:
  int _freq;
  uint8_t _low;
  uint8_t _high;
  double _duty;
};
}  // namespace autd::modulation
