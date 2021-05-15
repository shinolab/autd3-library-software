// File: gain.hpp
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

#include <memory>
#include <string>
#include <vector>

#include "geometry.hpp"
#include "hardware_defined.hpp"
#include "result.hpp"

namespace autd::core {

class Gain;
using GainPtr = std::shared_ptr<Gain>;

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
  [[nodiscard]] virtual Result<bool, std::string> Calc() {
    for (size_t i = 0; i < this->_geometry->num_devices(); i++) this->_data[i].fill(0x0000);
    return Ok(true);
  }

  /**
   * @brief Initialize data and calculate amplitude and phase of each transducer
   */
  [[nodiscard]] Result<bool, std::string> Build() {
    if (this->_built) return Ok(true);

    const auto num_device = this->_geometry->num_devices();

    this->_data.clear();
    this->_data.resize(num_device);

    auto res = this->Calc();
    if (res.is_err()) return res;

    this->_built = true;
    return Ok(res.unwrap());
  }

  /**
   * @brief Re-calculate amplitude and phase of each transducer
   */
  [[nodiscard]] Result<bool, std::string> Rebuild() {
    this->_built = false;
    return this->Build();
  }

  /**
   * @brief Set AUTD Geometry which is required to build gain
   */

  void SetGeometry(const GeometryPtr& geometry) noexcept { this->_geometry = geometry; }

  /**
   * @brief Get AUTD Geometry
   */
  [[nodiscard]] GeometryPtr geometry() const noexcept { return _geometry; }

  /**
   * @brief Getter function for the data of amplitude and phase of each transducers
   * @details Each data is 16 bit unsigned integer, where MSB represents amplitude and LSB represents phase
   */
  std::vector<AUTDDataArray>& data() { return _data; }

  Gain() noexcept : _built(false), _geometry(nullptr) {}
  virtual ~Gain() = default;
  Gain(const Gain& v) noexcept = default;
  Gain& operator=(const Gain& obj) = default;
  Gain(Gain&& obj) = default;
  Gain& operator=(Gain&& obj) = default;

 protected:
  bool _built;
  GeometryPtr _geometry;
  std::vector<AUTDDataArray> _data;
};  // namespace autd::gain
}  // namespace autd::core
