// File: modulation.hpp
// Project: core
// Created Date: 11/05/2021
// Author: Shun Suzuki
// -----
// Last Modified: 23/07/2021
// Modified By: Shun Suzuki (suzuki@hapis.k.u-tokyo.ac.jp)
// -----
// Copyright (c) 2021 Hapis Lab. All rights reserved.
//

#pragma once

#include <memory>
#include <string>
#include <vector>

#include "exception.hpp"
#include "hardware_defined.hpp"
#include "utils.hpp"

namespace autd::core {

class Modulation;
using ModulationPtr = std::shared_ptr<Modulation>;

/**
 * @brief Modulation controls the amplitude modulation
 */
class Modulation {
 public:
  Modulation() noexcept : Modulation(10) {}
  explicit Modulation(const uint16_t freq_div) noexcept : _freq_div(freq_div), _sent(0) {}
  virtual ~Modulation() = default;
  Modulation(const Modulation& v) noexcept = default;
  Modulation& operator=(const Modulation& obj) = default;
  Modulation(Modulation&& obj) = default;
  Modulation& operator=(Modulation&& obj) = default;

  /**
   * @brief Generate empty modulation, which produce static pressure
   * \param duty duty ratio
   */
  static ModulationPtr create(const uint8_t duty = 0xFF) {
    auto mod = std::make_shared<Modulation>();
    mod->_buffer.resize(1, duty);
    return mod;
  }

  /**
   * @brief Generate empty modulation, which produce static pressure
   * \param amp relative amplitude (0 to 1)
   */
  static ModulationPtr create(const double amp) { return create(Utilities::to_duty(amp)); }

  /**
   * \brief Calculate modulation data
   */
  virtual void calc() { return; }

  /**
   * \brief Build modulation data
   */
  void build() {
    this->calc();
    if (this->_buffer.size() > MOD_BUF_SIZE_MAX) throw core::ModulationBuildError("Modulation buffer overflow");
    return;
  }

  /**
   * \brief sent means data length already sent to devices.
   */
  size_t& sent() { return _sent; }

  /**
   * \brief modulation data
   */
  std::vector<uint8_t>& buffer() { return _buffer; }

  /**
   * \brief modulation sampling frequency division
   */
  [[nodiscard]] uint16_t sampling_frequency_division() const noexcept { return _freq_div; }

  /**
   * \brief modulation sampling frequency
   */
  [[nodiscard]] double sampling_freq() const noexcept { return static_cast<double>(MOD_SAMPLING_FREQ_BASE) / static_cast<double>(this->_freq_div); }

 protected:
  uint16_t _freq_div;
  size_t _sent;
  std::vector<uint8_t> _buffer;
};

}  // namespace autd::core
