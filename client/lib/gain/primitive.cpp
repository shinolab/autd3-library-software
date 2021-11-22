// File: primitive.cpp
// Project: gain
// Created Date: 14/04/2021
// Author: Shun Suzuki
// -----
// Last Modified: 22/11/2021
// Modified By: Shun Suzuki (suzuki@hapis.k.u-tokyo.ac.jp)
// -----
// Copyright (c) 2021 Hapis Lab. All rights reserved.
//

#include "autd3/gain/primitive.hpp"

#include <memory>
#include <vector>

#include "autd3/core/utils.hpp"

namespace autd::gain {

using core::Vector3;

std::shared_ptr<Grouped> Grouped::create() { return std::make_shared<Grouped>(); }

void Grouped::add(const size_t device_id, const GainPtr& gain) { this->_gain_map[device_id] = gain; }

void Grouped::calc(const core::Geometry& geometry) {
  for (const auto& [_, g] : this->_gain_map) g->build(geometry);
  for (size_t i = 0; i < geometry.num_devices(); i++)
    this->_data[i] = _gain_map.count(i) > 0 ? _gain_map[i]->data()[i] : core::GainData{core::Drive(0x00, 0x00)};
}

GainPtr PlaneWave::create(const Vector3& direction, const double amp) { return create(direction, core::utils::to_duty(amp)); }

GainPtr PlaneWave::create(const Vector3& direction, uint8_t duty) { return std::make_shared<PlaneWave>(direction, duty); }

void PlaneWave::calc(const core::Geometry& geometry) {
  const auto dir = this->_direction.normalized();

  const auto wavenum = 2.0 * M_PI / geometry.wavelength();
  for (const auto& dev : geometry)
    for (const auto& transducer : dev) {
      const auto dist = transducer.position().dot(dir);
      this->_data[dev.id()][transducer.id()].phase = core::utils::to_phase(dist * wavenum);
      this->_data[dev.id()][transducer.id()].duty = this->_duty;
    }
}

GainPtr FocalPoint::create(const Vector3& point, const double amp) { return create(point, core::utils::to_duty(amp)); }
GainPtr FocalPoint::create(const Vector3& point, uint8_t duty) { return std::make_shared<FocalPoint>(point, duty); }

void FocalPoint::calc(const core::Geometry& geometry) {
  const auto wavenum = 2.0 * M_PI / geometry.wavelength();
  for (const auto& dev : geometry)
    for (const auto& transducer : dev) {
      const auto dist = (transducer.position() - this->_point).norm();
      this->_data[dev.id()][transducer.id()].duty = this->_duty;
      this->_data[dev.id()][transducer.id()].phase = core::utils::to_phase(dist * wavenum);
    }
}

GainPtr BesselBeam::create(const Vector3& apex, const Vector3& vec_n, const double theta_z, const double amp) {
  return create(apex, vec_n, theta_z, core::utils::to_duty(amp));
}

GainPtr BesselBeam::create(const Vector3& apex, const Vector3& vec_n, double theta_z, uint8_t duty) {
  return std::make_shared<BesselBeam>(apex, vec_n, theta_z, duty);
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
      this->_data[dev.id()][transducer.id()].duty = this->_duty;
      this->_data[dev.id()][transducer.id()].phase = core::utils::to_phase(d * wavenum);
    }
}

GainPtr TransducerTest::create(const size_t transducer_index, const uint8_t duty, const uint8_t phase) {
  return std::make_shared<TransducerTest>(transducer_index, duty, phase);
}

void TransducerTest::calc(const core::Geometry& geometry) {
  size_t device_idx, local_trans_idx;
  core::Geometry::global_to_local_idx(_transducer_idx, &device_idx, &local_trans_idx);
  this->_data[device_idx][local_trans_idx].duty = this->_duty;
  this->_data[device_idx][local_trans_idx].phase = this->_phase;
}
}  // namespace autd::gain
