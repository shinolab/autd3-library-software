// File: modulation.hpp
// Project: core
// Created Date: 11/05/2021
// Author: Shun Suzuki
// -----
// Last Modified: 08/06/2021
// Modified By: Shun Suzuki (suzuki@hapis.k.u-tokyo.ac.jp)
// -----
// Copyright (c) 2021 Hapis Lab. All rights reserved.
//

#pragma once

#include <memory>
#include <vector>

#include "configuration.hpp"
#include "result.hpp"
#include "utils.hpp"

namespace autd::core {

class Modulation;
using ModulationPtr = std::shared_ptr<Modulation>;

/**
 * @brief Modulation controls the amplitude modulation
 */
class Modulation {
 public:
  Modulation() noexcept : _sent(0) {}
  virtual ~Modulation() = default;
  Modulation(const Modulation& v) noexcept = default;
  Modulation& operator=(const Modulation& obj) = default;
  Modulation(Modulation&& obj) = default;
  Modulation& operator=(Modulation&& obj) = default;

  /**
   * \brief Convert ultrasound amplitude to duty ratio.
   * \param amp ultrasound amplitude
   * \return duty ratio
   */
  static uint8_t to_duty(const double amp) noexcept { return Utilities::to_duty(amp); }

  /**
   * @brief Generate empty modulation, which produce static pressure
   * \param duty duty ratio
   */
  static ModulationPtr create(const uint8_t duty = 0xFF) {
    auto mod = std::make_shared<Modulation>();
    mod->_buffer.resize(MOD_FRAME_SIZE, duty);
    return mod;
  }

  /**
   * @brief Generate empty modulation, which produce static pressure
   * \param amp relative amplitude (0 to 1)
   */
  static ModulationPtr create(const double amp) { return create(to_duty(amp)); }

  /**
   * \brief Build modulation data with Configuration
   * \param config Configuration
   * \return ok(whether succeeded), or err(error message) if unrecoverable error is occurred
   */
  [[nodiscard]] virtual Error build(Configuration config) {
    (void)config;
    return Ok(true);
  }

  /**
   * \brief sent means data length already sent to devices.
   */
  size_t& sent() { return _sent; }

  /**
   * \brief modulation data
   */
  std::vector<uint8_t>& buffer() { return _buffer; }

 protected:
  size_t _sent;
  std::vector<uint8_t> _buffer;
};

}  // namespace autd::core
