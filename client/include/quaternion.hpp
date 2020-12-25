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

namespace autd {
namespace utils {
/**
 * @brief Simple quaternion class
 */
class Quaternion {
 public:
  /**
   * @brief Create Quaternion
   */
  Quaternion(const double w, const double x, const double y, const double z) {
    this->_x = x;
    this->_y = y;
    this->_z = z;
    this->_w = w;
  }

  [[nodiscard]] double x() const { return _x; }
  [[nodiscard]] double y() const { return _y; }
  [[nodiscard]] double z() const { return _z; }
  [[nodiscard]] double w() const { return _w; }

private:
    double _x;
    double _y;
    double _z;
    double _w;
};
}  // namespace _utils
}  // namespace autd
