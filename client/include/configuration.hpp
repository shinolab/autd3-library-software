// File: configuration.hpp
// Project: include
// Created Date: 30/10/2020
// Author: Shun Suzuki
// -----
// Last Modified: 30/10/2020
// Modified By: Shun Suzuki (suzuki@hapis.k.u-tokyo.ac.jp)
// -----
// Copyright (c) 2020 Hapis Lab. All rights reserved.
//

#pragma once

#include <iostream>
#include <string>
#include <vector>

namespace autd {

enum class MOD_SAMPLING_FREQ {
  SMPL_125_HZ = 125,
  SMPL_250_HZ = 250,
  SMPL_500_HZ = 500,
  SMPL_1_KHZ = 1000,
  SMPL_2_KHZ = 2000,
  SMPL_4_KHZ = 4000,
  SMPL_8_KHZ = 8000,
};

enum class MOD_BUF_SIZE {
  BUF_125 = 125,
  BUF_250 = 250,
  BUF_500 = 500,
  BUF_1000 = 1000,
  BUF_2000 = 2000,
  BUF_4000 = 4000,
  BUF_8000 = 8000,
  BUF_16000 = 16000,
  BUF_32000 = 32000,
};

/**
 * @brief AUTD Configuration
 */
class Configuration {
 public:
  static Configuration GetDefaultConfiguration() {
    Configuration config;
    return config;
  }

  void set_mod_sampling_freq(MOD_SAMPLING_FREQ f) { this->_mod_sampl_freq = f; }
  void set_mod_buf_size(MOD_BUF_SIZE s) { this->_mod_buf_size = s; }

  MOD_SAMPLING_FREQ mod_sampling_freq() { return this->_mod_sampl_freq; }
  MOD_BUF_SIZE mod_buf_size() { return this->_mod_buf_size; }

 private:
  Configuration() {
    _mod_sampl_freq = MOD_SAMPLING_FREQ::SMPL_4_KHZ;
    _mod_buf_size = MOD_BUF_SIZE::BUF_4000;
  }

  MOD_SAMPLING_FREQ _mod_sampl_freq;
  MOD_BUF_SIZE _mod_buf_size;
};
}  // namespace autd
