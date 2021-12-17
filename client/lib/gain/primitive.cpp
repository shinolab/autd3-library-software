// File: primitive.cpp
// Project: gain
// Created Date: 14/04/2021
// Author: Shun Suzuki
// -----
// Last Modified: 13/12/2021
// Modified By: Shun Suzuki (suzuki@hapis.k.u-tokyo.ac.jp)
// -----
// Copyright (c) 2021 Hapis Lab. All rights reserved.
//

#include "autd3/gain/primitive.hpp"

#include <vector>

#include "autd3/core/utils.hpp"

namespace autd::gain {

using core::Vector3;

void PlaneWave::calc(const core::Geometry& geometry) {
  const auto dir = this->_direction.normalized();

  const auto wavenum = 2.0 * M_PI / geometry.wavelength();
  for (const auto& dev : geometry)
    for (const auto& transducer : dev) {
      const auto dist = transducer.position().dot(dir);
      this->_data[transducer.id()].phase = core::utils::to_phase(dist * wavenum);
      this->_data[transducer.id()].duty = this->_duty;
    }
}

void FocalPoint::calc(const core::Geometry& geometry) {
  const auto wavenum = 2.0 * M_PI / geometry.wavelength();
  for (const auto& dev : geometry)
    for (const auto& transducer : dev) {
      const auto dist = (transducer.position() - this->_point).norm();
      this->_data[transducer.id()].duty = this->_duty;
      this->_data[transducer.id()].phase = core::utils::to_phase(dist * wavenum);
    }
}

void BesselBeam::calc(const core::Geometry& geometry) {
  _vec_n.normalize();
  Vector3 v = Vector3::UnitZ().cross(_vec_n);
  const auto theta_v = std::asin(v.norm());
  v.normalize();
  const Eigen::AngleAxisd rot(-theta_v, v);

  const auto wavenum = 2.0 * M_PI / geometry.wavelength();
  for (const auto& dev : geometry)
    for (const auto& transducer : dev) {
      const auto r = transducer.position() - this->_apex;
      const auto rr = rot * r;
      const auto d = std::sin(_theta_z) * std::sqrt(rr.x() * rr.x() + rr.y() * rr.y()) - std::cos(_theta_z) * rr.z();
      this->_data[transducer.id()].duty = this->_duty;
      this->_data[transducer.id()].phase = core::utils::to_phase(d * wavenum);
    }
}

void TransducerTest::calc(const core::Geometry& geometry) {
  this->_data[_transducer_idx].duty = this->_duty;
  this->_data[_transducer_idx].phase = this->_phase;
}
}  // namespace autd::gain
