// File: configuration.hpp
// Project: core
// Created Date: 14/04/2021
// Author: Shun Suzuki
// -----
// Last Modified: 19/05/2021
// Modified By: Shun Suzuki (suzuki@hapis.k.u-tokyo.ac.jp)
// -----
// Copyright (c) 2021 Hapis Lab. All rights reserved.
//

#pragma once

#include <algorithm>

#include "hardware_defined.hpp"

namespace autd::core {
/**
 * @brief AUTD Configuration
 */
class Configuration {
 public:
  Configuration(const uint16_t sampling_freq_div, const uint16_t buf_size) {
    _mod_sampling_freq_div = std::max(sampling_freq_div, uint16_t{1});
    _mod_buf_size = std::min(buf_size, MOD_BUF_SIZE_MAX);
  }

  static Configuration get_default_configuration() {
    const Configuration config(10, 4000);
    return config;
  }

  [[nodiscard]] double mod_sampling_freq() const noexcept {
    return static_cast<double>(MOD_SAMPLING_FREQ_BASE) / static_cast<double>(this->_mod_sampling_freq_div);
  }
  [[nodiscard]] uint16_t mod_sampling_freq_div() const noexcept { return this->_mod_sampling_freq_div; }
  [[nodiscard]] uint16_t mod_buf_size() const noexcept { return this->_mod_buf_size; }

 private:
  uint16_t _mod_sampling_freq_div;
  uint16_t _mod_buf_size;
};
}  // namespace autd::core
