// File: modulation.hpp
// Project: include
// Created Date: 04/11/2018
// Author: Shun Suzuki
// -----
// Last Modified: 03/04/2021
// Modified By: Shun Suzuki (suzuki@hapis.k.u-tokyo.ac.jp)
// -----
// Copyright (c) 2018-2020 Hapis Lab. All rights reserved.
//

#pragma once

#include <cmath>
#include <limits>
#include <memory>
#include <string>
#include <vector>

#include "autd_types.hpp"
#include "configuration.hpp"
#include "consts.hpp"
#include "result.hpp"

namespace autd {

namespace modulation {
class Modulation;
}

using ModulationPtr = std::shared_ptr<modulation::Modulation>;

namespace modulation {

inline Float Sinc(const Float x) noexcept {
  if (fabs(x) < std::numeric_limits<Float>::epsilon()) return 1;
  return std::sin(PI * x) / (PI * x);
}

/**
 * @brief Modulation controls the amplitude modulation
 */
class Modulation {
 public:
  Modulation() noexcept;
  virtual ~Modulation() = default;
  Modulation(const Modulation& v) noexcept = default;
  Modulation& operator=(const Modulation& obj) = default;
  Modulation(Modulation&& obj) = default;
  Modulation& operator=(Modulation&& obj) = default;

  /**
   * @brief Generate empty modulation, which produce static pressure
   */
  static ModulationPtr Create(uint8_t amp = 0xff);
  [[nodiscard]] virtual Result<bool, std::string> Build(Configuration config);
  std::vector<uint8_t> buffer;
  size_t& sent();

 private:
  size_t _sent;
};

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
  Result<bool, std::string> Build(Configuration config) override;
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
  Result<bool, std::string> Build(Configuration config) override;
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
  Result<bool, std::string> Build(Configuration config) override;
  explicit SawModulation(const int freq) : Modulation(), _freq(freq) {}

 private:
  int _freq = 0;
};
}  // namespace modulation
}  // namespace autd
