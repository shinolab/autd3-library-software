// File: modulation.hpp
// Project: include
// Created Date: 04/11/2018
// Author: Shun Suzuki
// -----
// Last Modified: 30/04/2020
// Modified By: Shun Suzuki (suzuki@hapis.k.u-tokyo.ac.jp)
// -----
// Copyright (c) 2018-2020 Hapis Lab. All rights reserved.
//

#pragma once

#include <memory>
#include <string>
#include <vector>

#include "core.hpp"

namespace autd {
namespace modulation {
/**
 * @brief Modulation controls the amplitude modulation
 */
class Modulation {
  friend class AUTDController;

 public:
  Modulation() noexcept;
  /**
   * @brief Genrate empty modulation, which produce static pressure
   */
  static ModulationPtr Create(uint8_t amp = 0xff);
  /**
   * @return int Sampling frequency of modulation
   */
  constexpr static int samplingFrequency();
  std::vector<uint8_t> buffer;

 private:
  size_t _sent;
};

/**
 * @brief Sine wave modulation
 */
class SineModulation : public Modulation {
 public:
  /**
   * @brief Generate function
   * @param[in] freq Frequency of the sine wave
   * @param[in] amp peek to peek amplitude of the wave (Maximum value is 1.0)
   * @param[in] offset offset of the wave
   * @details The sine wave oscillate from offset-amp/2 to offset+amp/2
   */
  static ModulationPtr Create(int freq, double amp = 1.0, double offset = 0.5);
};

/**
 * @brief Sawtooth wave modulation
 */
class SawModulation : public Modulation {
 public:
  /**
   * @brief Generate function
   * @param[in] freq Frequency of the sawtooth wave
   */
  static ModulationPtr Create(int freq);
};

/**
 * @brief Modulation created from raw pcm data
 */
class RawPCMModulation : public Modulation {
 public:
  /**
   * @brief Generate function
   * @param[in] filename file path to raw pcm data
   * @param[in] samplingFreq sampling frequency of the data
   * @details The sampling frequency of AUTD is shown in autd::MOD_SAMPLING_FREQ, and it is not possible to modulate beyond the Nyquist frequency.
   * No modulation beyond the Nyquist frequency can be produced.
   * If samplingFreq is less than the Nyquist frequency, the data will be upsampled.
   * The maximum modulation buffer size is shown in autd::MOD_BUF_SIZE. Only the data up to MOD_BUF_SIZE/MOD_SAMPLING_FREQ seconds can be output.
   */
  static ModulationPtr Create(std::string filename, double samplingFreq = 0.0);
};

/**
 * @brief Modulation created from wav file
 */
class WavModulation : public Modulation {
 public:
  /**
   * @brief Generate function
   * @param[in] filename file path to wav data
   * @details The sampling frequency of AUTD is shown in autd::MOD_SAMPLING_FREQ, and it is not possible to modulate beyond the Nyquist frequency.
   * No modulation beyond the Nyquist frequency can be produced.
   * If samplingFreq is less than the Nyquist frequency, the data will be upsampled.
   * The maximum modulation buffer size is shown in autd::MOD_BUF_SIZE. Only the data up to MOD_BUF_SIZE/MOD_SAMPLING_FREQ seconds can be output.
   */
  static ModulationPtr Create(std::string filename);
};
}  // namespace modulation
}  // namespace autd