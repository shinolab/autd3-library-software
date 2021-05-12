// File: configuration.hpp
// Project: core
// Created Date: 14/04/2021
// Author: Shun Suzuki
// -----
// Last Modified: 12/05/2021
// Modified By: Shun Suzuki (suzuki@hapis.k.u-tokyo.ac.jp)
// -----
// Copyright (c) 2021 Hapis Lab. All rights reserved.
//

#pragma once

#include "hardware_defined.hpp"

namespace autd::core {
/**
 * @brief AUTD Configuration
 */
class Configuration {
 public:
  Configuration() {
    _mod_sampling_freq = MOD_SAMPLING_FREQ::SMPL_4_KHZ;
    _mod_buf_size = MOD_BUF_SIZE::BUF_4000;
  }

  static Configuration GetDefaultConfiguration() {
    const Configuration config;
    return config;
  }

  MOD_SAMPLING_FREQ& mod_sampling_freq() noexcept { return this->_mod_sampling_freq; }
  MOD_BUF_SIZE& mod_buf_size() noexcept { return this->_mod_buf_size; }
  [[nodiscard]] MOD_SAMPLING_FREQ mod_sampling_freq() const noexcept { return this->_mod_sampling_freq; }
  [[nodiscard]] MOD_BUF_SIZE mod_buf_size() const noexcept { return this->_mod_buf_size; }

 private:
  MOD_SAMPLING_FREQ _mod_sampling_freq;
  MOD_BUF_SIZE _mod_buf_size;
};
}  // namespace autd::core
