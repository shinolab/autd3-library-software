// File: geometry.cpp
// Project: lib
// Created Date: 08/06/2016
// Author: Seki Inoue
// -----
// Last Modified: 26/12/2020
// Modified By: Shun Suzuki (suzuki@hapis.k.u-tokyo.ac.jp)
// -----
// Copyright (c) 2016-2020 Hapis Lab. All rights reserved.
//

#include <map>

#if WIN32
#include <codeanalysis/warnings.h>  // NOLINT
#pragma warning(push)
#pragma warning(disable : ALL_CODE_ANALYSIS_WARNINGS)
#endif
#include <Eigen/Geometry>
#if WIN32
#pragma warning(pop)
#endif

#include "autd3.hpp"
#include "autd_logic.hpp"
#include "convert.hpp"
#include "vector3.hpp"

using autd::IsMissingTransducer;
using autd::NUM_TRANS_IN_UNIT;
using autd::NUM_TRANS_X;
using autd::NUM_TRANS_Y;
using autd::TRANS_SIZE_MM;

namespace autd {
struct Device {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  static Device Create(const Eigen::Vector3d& position, const Eigen::Quaterniond& quaternion) {
    const Eigen::Affine3d transform_matrix = Eigen::Translation3d(position) * quaternion;
    const auto x_direction = quaternion * Eigen::Vector3d(1, 0, 0);
    const auto y_direction = quaternion * Eigen::Vector3d(0, 1, 0);
    const auto z_direction = quaternion * Eigen::Vector3d(0, 0, 1);

    Eigen::Matrix<double, 3, NUM_TRANS_IN_UNIT> local_trans_positions;

    auto index = 0;
    for (size_t y = 0; y < NUM_TRANS_Y; y++)
      for (size_t x = 0; x < NUM_TRANS_X; x++)
        if (!IsMissingTransducer(x, y))
          local_trans_positions.col(index++) = Eigen::Vector3d(static_cast<double>(x) * TRANS_SIZE_MM, static_cast<double>(y) * TRANS_SIZE_MM, 0);

    const auto global_trans_positions = transform_matrix * local_trans_positions;

    return Device{x_direction, y_direction, z_direction, global_trans_positions};
  }

  static Device Create(const Eigen::Vector3d& position, Eigen::Vector3d euler_angles) {
    const auto quaternion = Eigen::AngleAxis<double>(euler_angles.x(), Eigen::Vector3d::UnitZ()) *
                            Eigen::AngleAxis<double>(euler_angles.y(), Eigen::Vector3d::UnitY()) *
                            Eigen::AngleAxis<double>(euler_angles.z(), Eigen::Vector3d::UnitZ());

    return Create(position, quaternion);
  }

  Eigen::Vector3d x_direction;
  Eigen::Vector3d y_direction;
  Eigen::Vector3d z_direction;
  Eigen::Matrix<double, 3, NUM_TRANS_IN_UNIT> global_trans_positions;
};

class AUTDGeometry final : public Geometry {
 public:
  AUTDGeometry() : _wavelength(8.5) {}
  ~AUTDGeometry() override = default;
  AUTDGeometry(const AUTDGeometry& v) noexcept = default;
  AUTDGeometry& operator=(const AUTDGeometry& obj) = default;
  AUTDGeometry(AUTDGeometry&& obj) = default;
  AUTDGeometry& operator=(AUTDGeometry&& obj) = default;

  size_t AddDevice(utils::Vector3 position, utils::Vector3 euler_angles, size_t group = 0) override;
  size_t AddDeviceQuaternion(utils::Vector3 position, utils::Quaternion quaternion, size_t group = 0) override;
#ifdef USE_EIGEN_AUTD
  size_t AddDevice(Vector3 position, Vector3 euler_angles, size_t group = 0) override;
  size_t AddDeviceQuaternion(Vector3 position, Quaternion quaternion, size_t group = 0) override;
#endif

  double wavelength() noexcept override;
  void set_wavelength(double wavelength) noexcept override;

  size_t num_devices() noexcept override;
  size_t num_transducers() noexcept override;
  size_t group_id_for_device_idx(size_t device_idx) override;
  Vector3 position(size_t global_transducer_idx) override;
  Vector3 position(size_t device, size_t local_transducer_idx) override;
  Vector3 local_position(size_t device_idx, utils::Vector3 global_position) override;
#ifdef USE_EIGEN_AUTD
  Vector3 local_position(size_t device_idx, Vector3 global_position) override;
#endif
  Vector3 direction(size_t device_idx) override;
  Vector3 x_direction(size_t device_idx) override;
  Vector3 y_direction(size_t device_idx) override;
  Vector3 z_direction(size_t device_idx) override;
  size_t device_idx_for_trans_idx(size_t transducer_idx) override;

 private:
  std::vector<Device> _devices;
  std::map<size_t, size_t> _group_map;
  double _wavelength;
};

GeometryPtr Geometry::Create() { return std::make_shared<AUTDGeometry>(); }

size_t AUTDGeometry::AddDevice(const utils::Vector3 position, const utils::Vector3 euler_angles, const size_t group) {
  const auto device_id = this->_devices.size();
  this->_devices.push_back(Device::Create(ConvertToEigen(position), ConvertToEigen(euler_angles)));
  this->_group_map[device_id] = group;
  return device_id;
}

size_t AUTDGeometry::AddDeviceQuaternion(const utils::Vector3 position, const utils::Quaternion quaternion, const size_t group) {
  const auto device_id = this->_devices.size();
  this->_devices.push_back(Device::Create(ConvertToEigen(position), ConvertToEigen(quaternion)));
  this->_group_map[device_id] = group;
  return device_id;
}
#ifdef USE_EIGEN_AUTD
size_t AUTDGeometry::AddDevice(const Vector3 position, const Vector3 euler_angles, const size_t group) {
  const auto device_id = this->_devices.size();
  this->_devices.push_back(Device::Create(position, euler_angles));
  this->_group_map[device_id] = group;
  return device_id;
}
size_t AUTDGeometry::AddDeviceQuaternion(const Vector3 position, const Quaternion quaternion, const size_t group) {
  const auto device_id = this->_devices.size();
  this->_devices.push_back(Device::Create(position, quaternion));
  this->_group_map[device_id] = group;
  return device_id;
}
#endif

double AUTDGeometry::wavelength() noexcept { return this->_wavelength; }
void AUTDGeometry::set_wavelength(const double wavelength) noexcept { this->_wavelength = wavelength; }

size_t AUTDGeometry::num_devices() noexcept { return this->_devices.size(); }

size_t AUTDGeometry::num_transducers() noexcept { return this->num_devices() * NUM_TRANS_IN_UNIT; }

size_t AUTDGeometry::group_id_for_device_idx(const size_t device_idx) { return this->_group_map[device_idx]; }

Vector3 AUTDGeometry::position(const size_t global_transducer_idx) {
  const auto local_trans_id = global_transducer_idx % NUM_TRANS_IN_UNIT;
  return position(this->device_idx_for_trans_idx(global_transducer_idx), local_trans_id);
}

Vector3 AUTDGeometry::position(const size_t device, const size_t local_transducer_idx) {
  const auto& dev = this->_devices[device];
  const Eigen::Vector3d pos = dev.global_trans_positions.col(local_transducer_idx);
  return Convert(pos);
}

Vector3 AUTDGeometry::local_position(const size_t device_idx, const utils::Vector3 global_position) {
  const auto& device = this->_devices[device_idx];
  const auto& local_origin = device.global_trans_positions.col(0);
  const auto& x_dir = device.x_direction;
  const auto& y_dir = device.y_direction;
  const auto& z_dir = device.z_direction;
  const auto gp = ConvertToEigen(global_position);
  const auto rv = gp - local_origin;
  return Vector3(rv.dot(x_dir), rv.dot(y_dir), rv.dot(z_dir));
}

#ifdef USE_EIGEN_AUTD
Vector3 AUTDGeometry::local_position(const size_t device_idx, const Vector3 global_position) {
  const auto& device = this->_devices[device_idx];
  const auto local_origin = device.global_trans_positions.col(0);
  const auto& x_dir = device.x_direction;
  const auto& y_dir = device.y_direction;
  const auto& z_dir = device.z_direction;
  const auto rv = global_position - local_origin;
  return Vector3(rv.dot(x_dir), rv.dot(y_dir), rv.dot(z_dir));
}
#endif

Vector3 AUTDGeometry::direction(const size_t device_idx) { return z_direction(device_idx); }

Vector3 AUTDGeometry::x_direction(const size_t device_idx) {
  const auto& dir = this->_devices[device_idx].x_direction;
  return Convert(dir);
}

Vector3 AUTDGeometry::y_direction(const size_t device_idx) {
  const auto& dir = this->_devices[device_idx].x_direction;
  return Convert(dir);
}

Vector3 AUTDGeometry::z_direction(const size_t device_idx) {
  const auto& dir = this->_devices[device_idx].x_direction;
  return Convert(dir);
}

size_t AUTDGeometry::device_idx_for_trans_idx(const size_t transducer_idx) { return transducer_idx / NUM_TRANS_IN_UNIT; }
}  // namespace autd
