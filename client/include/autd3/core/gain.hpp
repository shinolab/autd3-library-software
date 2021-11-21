// File: gain.hpp
// Project: core
// Created Date: 11/05/2021
// Author: Shun Suzuki
// -----
// Last Modified: 21/11/2021
// Modified By: Shun Suzuki (suzuki@hapis.k.u-tokyo.ac.jp)
// -----
// Copyright (c) 2021 Hapis Lab. All rights reserved.
//

#pragma once

#include <memory>
#include <vector>

#include "geometry.hpp"
#include "hardware_defined.hpp"
#include "utils.hpp"

namespace autd::core {

class Gain;
using GainPtr = std::shared_ptr<Gain>;

/**
 * @brief Gain controls the duty ratio and phase of each transducer in AUTD devices.
 */
class Gain {
 public:
  /**
   * @brief Generate empty gain
   */
  static GainPtr create() { return std::make_shared<Gain>(); }

  /**
   * \brief Calculate duty ratio and phase of each transducer
   * \param geometry Geometry
   */
  virtual void calc(const Geometry& geometry) {
    for (size_t i = 0; i < geometry.num_devices(); i++) this->_data[i].fill(0x0000);
  }

  /**
   * \brief Initialize data and call calc().
   * \param geometry Geometry
   */
  void build(const Geometry& geometry) {
    if (this->_built) return;

    const auto num_device = geometry.num_devices();

    this->_data.clear();
    this->_data.resize(num_device);

    this->calc(geometry);
    this->_built = true;
  }

  /**
   * \brief Re-calculate duty ratio and phase of each transducer
   * \param geometry Geometry
   */
  void rebuild(const Geometry& geometry) {
    this->_built = false;
    this->build(geometry);
  }

  /**
   * @brief Getter function for the data of duty ratio and phase of each transducers
   * @details Each data is 16 bit unsigned integer, where high 8bits represents duty ratio and low 8bits represents phase
   */
  const std::vector<DataArray>& data() const { return _data; }

  Gain() noexcept : _built(false) {}
  virtual ~Gain() = default;
  Gain(const Gain& v) noexcept = default;
  Gain& operator=(const Gain& obj) = default;
  Gain(Gain&& obj) = default;
  Gain& operator=(Gain&& obj) = default;

 protected:
  bool _built;
  std::vector<DataArray> _data;
};
}  // namespace autd::core
