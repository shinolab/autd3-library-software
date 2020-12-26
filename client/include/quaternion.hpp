// File: quaternion.hpp
// Project: include
// Created Date: 27/02/2020
// Author: Shun Suzuki
// -----
// Last Modified: 26/12/2020
// Modified By: Shun Suzuki (suzuki@hapis.k.u-tokyo.ac.jp)
// -----
// Copyright (c) 2020 Hapis Lab. All rights reserved.
//

#pragma once

#include "autd_types.hpp"

namespace autd::utils {
/**
 * @brief Simple quaternion class
 */
class Quaternion {
 public:
  /**
   * @brief Create Quaternion
   */
  Quaternion(const Float w, const Float x, const Float y, const Float z) {
    this->_x = x;
    this->_y = y;
    this->_z = z;
    this->_w = w;
  }

  [[nodiscard]] Float x() const { return _x; }
  [[nodiscard]] Float y() const { return _y; }
  [[nodiscard]] Float z() const { return _z; }
  [[nodiscard]] Float w() const { return _w; }

 private:
  Float _x;
  Float _y;
  Float _z;
  Float _w;
};
}  // namespace autd::utils
