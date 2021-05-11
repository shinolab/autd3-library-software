// File: modulation.hpp
// Project: core
// Created Date: 11/05/2021
// Author: Shun Suzuki
// -----
// Last Modified: 11/05/2021
// Modified By: Shun Suzuki (suzuki@hapis.k.u-tokyo.ac.jp)
// -----
// Copyright (c) 2021 Hapis Lab. All rights reserved.
//

#pragma once

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
   * @brief Generate empty modulation, which produce static pressure
   */
  static ModulationPtr Create(uint8_t amp = 0xff) {
    auto mod = std::make_shared<Modulation>();
    mod->_buffer.resize(1, amp);
    return mod;
  }

  [[nodiscard]] virtual Result<bool, std::string> Build(Configuration config) {
    (void)config;
    return Ok(true);
  }

  size_t& sent() { return _sent; }

 protected:
  size_t _sent;
  std::vector<uint8_t> _buffer;
};

}  // namespace autd::core
