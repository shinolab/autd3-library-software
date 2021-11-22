// File: gain.hpp
// Project: core
// Created Date: 11/05/2021
// Author: Shun Suzuki
// -----
// Last Modified: 22/11/2021
// Modified By: Shun Suzuki (suzuki@hapis.k.u-tokyo.ac.jp)
// -----
// Copyright (c) 2021 Hapis Lab. All rights reserved.
//

#pragma once

#include <memory>
#include <vector>

#include "geometry.hpp"
#include "hardware_defined.hpp"

namespace autd {
namespace core {

class Gain;
using GainPtr = std::shared_ptr<Gain>;

struct Drive final {
  Drive() : Drive(0x00, 0x00) {}
  explicit Drive(const uint8_t duty, const uint8_t phase) : phase(phase), duty(duty) {}

  uint8_t phase;
  uint8_t duty;
};

using GainData = std::array<Drive, NUM_TRANS_IN_UNIT>;

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
    for (size_t i = 0; i < geometry.num_devices(); i++) this->_data[i].fill(Drive(0x00, 0x00));
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
   */
  [[nodiscard]] const std::vector<GainData>& data() const { return _data; }

  Gain() noexcept : _built(false) {}
  virtual ~Gain() = default;
  Gain(const Gain& v) noexcept = default;
  Gain& operator=(const Gain& obj) = default;
  Gain(Gain&& obj) = default;
  Gain& operator=(Gain&& obj) = default;

 protected:
  bool _built;
  std::vector<GainData> _data;
};
}  // namespace core
}  // namespace autd
