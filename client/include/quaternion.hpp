// File: quaternion.hpp
// Project: include
// Created Date: 27/02/2020
// Author: Shun Suzuki
// -----
// Last Modified: 30/04/2020
// Modified By: Shun Suzuki (suzuki@hapis.k.u-tokyo.ac.jp)
// -----
// Copyright (c) 2020 Hapis Lab. All rights reserved.
//

#pragma once

#include <cmath>

#include "core.hpp"

namespace autd {
namespace _utils {
/**
 * @brief Simple quaternion class
 */
class Quaternion {
 private:
  double _x;
  double _y;
  double _z;
  double _w;

 public:
  /**
   * @brief Create Quaternion
   */
  Quaternion(double w, double x, double y, double z) {
    this->_x = x;
    this->_y = y;
    this->_z = z;
    this->_w = w;
  }

  double x() { return _x; }
  double y() { return _y; }
  double z() { return _z; }
  double w() { return _w; }
};
}  // namespace _utils
}  // namespace autd
