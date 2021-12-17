// File: primitive_modulation.hpp
// Project: include
// Created Date: 14/04/2021
// Author: Shun Suzuki
// -----
// Last Modified: 16/12/2021
// Modified By: Shun Suzuki (suzuki@hapis.k.u-tokyo.ac.jp)
// -----
// Copyright (c) 2021 Hapis Lab. All rights reserved.
//

#pragma once

#include "autd3/core/modulation.hpp"
#include "autd3/core/utils.hpp"

namespace autd::modulation {

using core::Modulation;

/**
 * @brief Modulation to avoid amplitude modulation
 */
class Static final : public Modulation {
 public:
  /**
   * \param duty duty ratio
   */
  explicit Static(const uint8_t duty = 0xFF) : Modulation() { this->_buffer.resize(1, duty); }

  void calc() override {}

  ~Static() override = default;
  Static(const Static& v) noexcept = delete;
  Static& operator=(const Static& obj) = delete;
  Static(Static&& obj) = default;
  Static& operator=(Static&& obj) = default;
};

/**
 * @brief Sine wave modulation in ultrasound amplitude
 */
class Sine final : public Modulation {
 public:
  /**
   * @param[in] freq Frequency of the sine wave
   * @param[in] amp peek to peek ultrasound amplitude (Maximum value is 1.0)
   * @param[in] offset offset of ultrasound amplitude
   * @details Ultrasound amplitude oscillate from offset-amp/2 to offset+amp/2.
   * If the value exceeds the range of [0, 1], the value will be clamped in the [0, 1].
   */
  explicit Sine(const int freq, const double amp = 1.0, const double offset = 0.5) : Modulation(), _freq(freq), _amp(amp), _offset(offset) {}

  void calc() override;

  ~Sine() override = default;
  Sine(const Sine& v) noexcept = delete;
  Sine& operator=(const Sine& obj) = delete;
  Sine(Sine&& obj) = default;
  Sine& operator=(Sine&& obj) = default;

 private:
  int _freq;
  double _amp;
  double _offset;
};

/**
 * @brief Sine wave modulation in squared acoustic pressure, which is proportional to radiation pressure
 */
class SineSquared final : public Modulation {
 public:
  /**
   * @param[in] freq Frequency of the sine wave
   * @param[in] amp peek to peek amplitude of radiation pressure (Maximum value is 1.0)
   * @param[in] offset offset of radiation pressure
   * @details Radiation pressure oscillate from offset-amp/2 to offset+amp/2
   * If the value exceeds the range of [0, 1], the value will be clamped in the [0, 1].
   */
  explicit SineSquared(const int freq, const double amp = 1.0, const double offset = 0.5) : Modulation(), _freq(freq), _amp(amp), _offset(offset) {}

  void calc() override;

  ~SineSquared() override = default;
  SineSquared(const SineSquared& v) noexcept = delete;
  SineSquared& operator=(const SineSquared& obj) = delete;
  SineSquared(SineSquared&& obj) = default;
  SineSquared& operator=(SineSquared&& obj) = default;

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
   * @param[in] freq Frequency of the sine wave
   * @param[in] amp peek to peek ultrasound amplitude (Maximum value is 1.0)
   * @param[in] offset offset of ultrasound amplitude
   * @details Ultrasound amplitude oscillate from offset-amp/2 to offset+amp/2.
   * If the value exceeds the range of [0, 1], the value will be clamped in the [0, 1].
   */
  explicit SineLegacy(const double freq, const double amp = 1.0, const double offset = 0.5) : Modulation(), _freq(freq), _amp(amp), _offset(offset) {}

  void calc() override;

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
   * @param[in] freq Frequency of the square wave
   * @param[in] low low level in duty
   * @param[in] high high level in duty
   * @param[in] duty duty ratio of square wave
   */
  Square(const int freq, const uint8_t low, const uint8_t high, const double duty = 0.5)
      : Modulation(), _freq(freq), _low(low), _high(high), _duty(duty) {}

  /**
   * @param[in] freq Frequency of the square wave
   * @param[in] low low level in relative amplitude (0 to 1)
   * @param[in] high high level in relative amplitude (0 to 1)
   * @param[in] duty duty ratio of square wave
   */
  Square(const int freq, const double low, const double high, const double duty = 0.5)
      : Square(freq, core::utils::to_duty(low), core::utils::to_duty(high), duty) {}

  void calc() override;

 private:
  int _freq;
  uint8_t _low;
  uint8_t _high;
  double _duty;
};
}  // namespace autd::modulation
