// File: from_file.hpp
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
#include <string>
#include <utility>
#include <vector>

#include "modulation.hpp"

namespace autd::modulation {
/**
 * @brief Modulation created from raw pcm data
 */
class RawPCMModulation final : public Modulation {
 public:
  /**
   * @brief Generate function
   * @param[in] filename file path to raw pcm data
   * @param[in] sampling_freq sampling frequency of the data
   * @details The sampling frequency of AUTD is shown in autd::MOD_SAMPLING_FREQ, and it is not possible to modulate beyond the Nyquist frequency.
   * No modulation beyond the Nyquist frequency can be produced.
   * If samplingFreq is less than the Nyquist frequency, the data will be upsampled.
   * The maximum modulation buffer size is shown in autd::MOD_BUF_SIZE. Only the data up to MOD_BUF_SIZE/MOD_SAMPLING_FREQ seconds can be output.
   */
  static ModulationPtr Create(const std::string& filename, Float sampling_freq = 0.0);
  void Build(Configuration config) override;
  explicit RawPCMModulation(const Float sampling_freq, std::vector<int32_t> buf)
      : Modulation(), _sampling_freq(sampling_freq), _buf(std::move(buf)) {}

 private:
  Float _sampling_freq = 0;
  std::vector<int32_t> _buf;
};

/**
 * @brief Modulation created from wav file
 */
class WavModulation final : public Modulation {
 public:
  /**
   * @brief Generate function
   * @param[in] filename file path to wav data
   * @details The sampling frequency of AUTD is shown in autd::MOD_SAMPLING_FREQ, and it is not possible to modulate beyond the Nyquist frequency.
   * No modulation beyond the Nyquist frequency can be produced.
   * If samplingFreq is less than the Nyquist frequency, the data will be upsampled.
   * The maximum modulation buffer size is shown in autd::MOD_BUF_SIZE. Only the data up to MOD_BUF_SIZE/MOD_SAMPLING_FREQ seconds can be output.
   */
  static ModulationPtr Create(const std::string& filename);
  void Build(Configuration config) override;
  explicit WavModulation(const uint32_t sampling_freq, std::vector<uint8_t> buf)
      : Modulation(), _sampling_freq(sampling_freq), _buf(std::move(buf)) {}

 private:
  uint32_t _sampling_freq = 0;
  std::vector<uint8_t> _buf;
};
}  // namespace autd::modulation
