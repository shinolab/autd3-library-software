// File: geometry.cpp
// Project: lib
// Created Date: 08/06/2016
// Author: Seki Inoue
// -----
// Last Modified: 25/12/2020
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
#include "controller.hpp"
#include "vector3.hpp"

using autd::IsMissingTransducer;
using autd::NUM_TRANS_IN_UNIT;
using autd::NUM_TRANS_X;
using autd::NUM_TRANS_Y;
using autd::TRANS_SIZE_MM;

namespace autd {

class Device {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  static std::unique_ptr<Device> Create(const Eigen::Vector3d& position, const Eigen::Quaterniond& quaternion) {
    auto device = std::make_unique<Device>();

    const Eigen::Affine3d transform_matrix = Eigen::Translation3d(position) * quaternion;
    device->x_direction = quaternion * Eigen::Vector3d(1, 0, 0);
    device->y_direction = quaternion * Eigen::Vector3d(0, 1, 0);
    device->z_direction = quaternion * Eigen::Vector3d(0, 0, 1);

    Eigen::Matrix<double, 3, NUM_TRANS_IN_UNIT> local_trans_positions;

    auto index = 0;
    for (size_t y = 0; y < NUM_TRANS_Y; y++)
      for (size_t x = 0; x < NUM_TRANS_X; x++)
        if (!IsMissingTransducer(x, y))
          local_trans_positions.col(index++) = Eigen::Vector3d(static_cast<double>(x) * TRANS_SIZE_MM, static_cast<double>(y) * TRANS_SIZE_MM, 0);

    device->global_trans_positions = transform_matrix * local_trans_positions;

    return device;
  }

  static std::unique_ptr<Device> Create(const Eigen::Vector3d& position, Eigen::Vector3d euler_angles) {
    const auto quaternion = Eigen::AngleAxis<double>(euler_angles.x(), Eigen::Vector3d::UnitZ()) *
                            Eigen::AngleAxis<double>(euler_angles.y(), Eigen::Vector3d::UnitY()) *
                            Eigen::AngleAxis<double>(euler_angles.z(), Eigen::Vector3d::UnitZ());

    return Create(position, quaternion);
  }

  Eigen::Matrix<double, 3, NUM_TRANS_IN_UNIT> global_trans_positions;
  Eigen::Vector3d x_direction;
  Eigen::Vector3d y_direction;
  Eigen::Vector3d z_direction;
};

class AUTDGeometry final : public Geometry {
 public:
  AUTDGeometry() = default;
  ~AUTDGeometry() override = default;
  AUTDGeometry(const AUTDGeometry& v) noexcept = default;
  AUTDGeometry& operator=(const AUTDGeometry& obj) = default;
  AUTDGeometry(AUTDGeometry&& obj) = default;
  AUTDGeometry& operator=(AUTDGeometry&& obj) = default;

  size_t AddDevice(Vector3 position, Vector3 euler_angles, size_t group = 0) override;
  size_t AddDeviceQuaternion(Vector3 position, Quaternion quaternion, size_t group = 0) override;

  size_t num_devices() noexcept override;
  size_t num_transducers() noexcept override;
  size_t group_id_for_device_idx(size_t device_idx) override;
  Vector3 position(size_t transducer_idx) override;
  Vector3 local_position(size_t device_idx, Vector3 global_position) override;
  Vector3 direction(size_t transducer_idx) override;
  Vector3 x_direction(size_t transducer_idx) override;
  Vector3 y_direction(size_t transducer_idx) override;
  Vector3 z_direction(size_t transducer_idx) override;
  size_t device_idx_for_trans_idx(size_t transducer_idx) override;

 private:
  std::vector<std::unique_ptr<Device>> _devices;
  std::map<size_t, size_t> _group_map;
};

GeometryPtr Geometry::Create() { return std::make_shared<AUTDGeometry>(); }

size_t AUTDGeometry::AddDevice(const Vector3 position, const Vector3 euler_angles, const size_t group) {
  const auto device_id = this->_devices.size();
  const auto pos = Eigen::Vector3d(position.x(), position.y(), position.z());
  const auto ea = Eigen::Vector3d(euler_angles.x(), euler_angles.y(), euler_angles.z());
  this->_devices.push_back(Device::Create(pos, ea));
  this->_group_map[device_id] = group;
  return device_id;
}

size_t AUTDGeometry::AddDeviceQuaternion(const Vector3 position, const Quaternion quaternion, const size_t group) {
  const auto device_id = this->_devices.size();
  const auto pos = Eigen::Vector3d(position.x(), position.y(), position.z());
  const auto qua = Eigen::Quaterniond(quaternion.w(), quaternion.x(), quaternion.y(), quaternion.z());
  this->_devices.push_back(Device::Create(pos, qua));
  this->_group_map[device_id] = group;
  return device_id;
}

size_t AUTDGeometry::num_devices() noexcept { return this->_devices.size(); }

size_t AUTDGeometry::num_transducers() noexcept { return this->num_devices() * NUM_TRANS_IN_UNIT; }

size_t AUTDGeometry::group_id_for_device_idx(const size_t device_idx) { return this->_group_map[device_idx]; }

Vector3 AUTDGeometry::position(const size_t transducer_idx) {
  const auto local_trans_id = transducer_idx % NUM_TRANS_IN_UNIT;
  const auto pos = this->_devices.at(this->device_idx_for_trans_idx(transducer_idx))->global_trans_positions.col(local_trans_id);
  return Vector3(pos.x(), pos.y(), pos.z());
}

Vector3 AUTDGeometry::local_position(const size_t device_idx, const Vector3 global_position) {
  const Device* device = this->_devices.at(device_idx).get();
  const auto local_origin = device->global_trans_positions.col(0);
  const auto& x_dir = device->x_direction;
  const auto& y_dir = device->y_direction;
  const auto& z_dir = device->z_direction;
  const auto gp = Eigen::Vector3d(global_position.x(), global_position.y(), global_position.z());
  const auto rv = gp - local_origin;
  return Vector3(rv.dot(x_dir), rv.dot(y_dir), rv.dot(z_dir));
}

Vector3 AUTDGeometry::direction(const size_t transducer_idx) { return z_direction(transducer_idx); }

Vector3 AUTDGeometry::x_direction(const size_t transducer_idx) {
  const auto& dir = this->_devices.at(this->device_idx_for_trans_idx(transducer_idx))->x_direction;
  return Vector3(dir.x(), dir.y(), dir.z());
}

Vector3 AUTDGeometry::y_direction(const size_t transducer_idx) {
  const auto& dir = this->_devices.at(this->device_idx_for_trans_idx(transducer_idx))->y_direction;
  return Vector3(dir.x(), dir.y(), dir.z());
}

Vector3 AUTDGeometry::z_direction(const size_t transducer_idx) {
  const auto& dir = this->_devices.at(this->device_idx_for_trans_idx(transducer_idx))->z_direction;
  return Vector3(dir.x(), dir.y(), dir.z());
}

size_t AUTDGeometry::device_idx_for_trans_idx(const size_t transducer_idx) { return transducer_idx / NUM_TRANS_IN_UNIT; }
}  // namespace autd
