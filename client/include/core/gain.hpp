// File: gain.hpp
// Project: core
// Created Date: 11/05/2021
// Author: Shun Suzuki
// -----
// Last Modified: 16/05/2021
// Modified By: Shun Suzuki (suzuki@hapis.k.u-tokyo.ac.jp)
// -----
// Copyright (c) 2021 Hapis Lab. All rights reserved.
//

#pragma once

#include <memory>
#include <string>
#include <vector>

#include "geometry.hpp"
#include "hardware_defined.hpp"
#include "result.hpp"

namespace autd::core {

class Gain;
using GainPtr = std::shared_ptr<Gain>;

template <typename T>
uint8_t ToDuty(const T amp) noexcept {
  const auto d = std::asin(amp) / static_cast<T>(M_PI);  //  duty (0 ~ 0.5)
  return static_cast<uint8_t>(511 * d);
}

/**
 * @brief Gain controls the amplitude and phase of each transducer in the AUTD
 */
class Gain {
 public:
  /**
   * @brief Generate empty gain
   */
  static GainPtr Create() { return std::make_shared<Gain>(); }

  /**
   * @brief Calculate amplitude and phase of each transducer
   */
  [[nodiscard]] virtual Result<bool, std::string> Calc(const GeometryPtr& geometry) {
    for (size_t i = 0; i < geometry->num_devices(); i++) this->_data[i].fill(0x0000);
    return Ok(true);
  }

  /**
   * @brief Initialize data and calculate amplitude and phase of each transducer
   */
  [[nodiscard]] Result<bool, std::string> Build(const GeometryPtr& geometry) {
    if (this->_built) return Ok(true);

    const auto num_device = geometry->num_devices();

    this->_data.clear();
    this->_data.resize(num_device);

    auto res = this->Calc(geometry);
    this->_built = res.unwrap_or(false);
    return res;
  }

  /**
   * @brief Re-calculate amplitude and phase of each transducer
   */
  [[nodiscard]] Result<bool, std::string> Rebuild(const GeometryPtr& geometry) {
    this->_built = false;
    return this->Build(geometry);
  }

  /**
   * @brief Getter function for the data of amplitude and phase of each transducers
   * @details Each data is 16 bit unsigned integer, where MSB represents amplitude and LSB represents phase
   */
  std::vector<AUTDDataArray>& data() { return _data; }

  Gain() noexcept : _built(false) {}
  virtual ~Gain() = default;
  Gain(const Gain& v) noexcept = default;
  Gain& operator=(const Gain& obj) = default;
  Gain(Gain&& obj) = default;
  Gain& operator=(Gain&& obj) = default;

 protected:
  bool _built;
  std::vector<AUTDDataArray> _data;
};  // namespace autd::gain
}  // namespace autd::core
