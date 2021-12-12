// File: gain.hpp
// Project: core
// Created Date: 11/05/2021
// Author: Shun Suzuki
// -----
// Last Modified: 12/12/2021
// Modified By: Shun Suzuki (suzuki@hapis.k.u-tokyo.ac.jp)
// -----
// Copyright (c) 2021 Hapis Lab. All rights reserved.
//

#pragma once

#include <vector>

#include "geometry.hpp"
#include "hardware_defined.hpp"
#include "interface.hpp"

namespace autd {
namespace core {

/**
 * @brief Gain controls the duty ratio and phase of each transducer in AUTD devices.
 */
class Gain : public IDatagramBody {
 public:
  /**
   * \brief Calculate duty ratio and phase of each transducer
   * \param geometry Geometry
   */
  virtual void calc(const Geometry& geometry) = 0;

  /**
   * \brief Initialize data and call calc().
   * \param geometry Geometry
   */
  void build(const Geometry& geometry) {
    if (this->_built) return;

    this->_data.clear();
    this->_data.resize(geometry.num_transducers());

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
  [[nodiscard]] const std::vector<Drive>& data() const { return _data; }

  void init() override {}

  uint8_t pack(const Geometry& geometry, TxDatagram& tx, uint8_t&& fpga_ctrl_flag, uint8_t&& cpu_ctrl_flag) override {
    this->build(geometry);

    const auto msg_id = get_id();
    fpga_ctrl_flag |= OUTPUT_ENABLE;
    cpu_ctrl_flag |= WRITE_BODY;
    std::memcpy(tx.data(), _data.data(), _data.size() * sizeof(Drive));
    return msg_id;
  }

  [[nodiscard]] bool is_finished() const override { return true; }

  Gain() noexcept : _built(false) {}
  ~Gain() override = default;
  Gain(const Gain& v) noexcept = delete;
  Gain& operator=(const Gain& obj) = delete;
  Gain(Gain&& obj) = default;
  Gain& operator=(Gain&& obj) = default;

 protected:
  bool _built;
  std::vector<Drive> _data;
};
}  // namespace core
}  // namespace autd
