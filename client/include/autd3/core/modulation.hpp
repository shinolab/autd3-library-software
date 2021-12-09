// File: modulation.hpp
// Project: core
// Created Date: 11/05/2021
// Author: Shun Suzuki
// -----
// Last Modified: 09/12/2021
// Modified By: Shun Suzuki (suzuki@hapis.k.u-tokyo.ac.jp)
// -----
// Copyright (c) 2021 Hapis Lab. All rights reserved.
//

#pragma once

#include <memory>
#include <vector>

#include "exception.hpp"
#include "hardware_defined.hpp"

namespace autd::core {

/**
 * @brief Modulation controls the amplitude modulation
 */
class Modulation {
 public:
  Modulation() noexcept : Modulation(10) {}
  explicit Modulation(const size_t freq_div) noexcept : _built(false), _freq_div_ratio(freq_div) {}
  virtual ~Modulation() = default;
  Modulation(const Modulation& v) noexcept = default;
  Modulation& operator=(const Modulation& obj) = default;
  Modulation(Modulation&& obj) = default;
  Modulation& operator=(Modulation&& obj) = default;

  /**
   * \brief Calculate modulation data
   */
  virtual void calc() = 0;

  /**
   * \brief Build modulation data
   */
  void build() {
    if (this->_built) return;
    if (_freq_div_ratio > MOD_SAMPLING_FREQ_DIV_MAX)
      throw exception::ModulationBuildError("Modulation sampling frequency division ratio is out of range");
    this->calc();
    if (this->_buffer.size() > MOD_BUF_SIZE_MAX) throw exception::ModulationBuildError("Modulation buffer overflow");
    this->_built = true;
  }

  /**
   * \brief Re-build modulation data
   */
  void rebuild() {
    this->_built = false;
    this->build();
  }

  /**
   * \brief modulation data
   */
  [[nodiscard]] const std::vector<uint8_t>& buffer() const { return _buffer; }

  /**
   * \brief sampling frequency division ratio
   * \details sampling frequency will be autd::core::MOD_SAMPLING_FREQ_BASE /(sampling frequency division ratio). The value must be in 1, 2, ...,
   * autd::core::MOD_SAMPLING_FREQ_DIV_MAX.
   */
  size_t& sampling_freq_div_ratio() noexcept { return _freq_div_ratio; }

  /**
   * \brief sampling frequency division ratio
   * \details sampling frequency will be autd::core::MOD_SAMPLING_FREQ_BASE /(sampling frequency division ratio). The value must be in 1, 2, ...,
   * autd::core::MOD_SAMPLING_FREQ_DIV_MAX.
   */
  size_t sampling_freq_div_ratio() const noexcept { return _freq_div_ratio; }

  /**
   * \brief modulation sampling frequency
   */
  [[nodiscard]] double sampling_freq() const noexcept { return static_cast<double>(MOD_SAMPLING_FREQ_BASE) / static_cast<double>(_freq_div_ratio); }

 protected:
  bool _built;
  size_t _freq_div_ratio;
  std::vector<uint8_t> _buffer;
};

}  // namespace autd::core
