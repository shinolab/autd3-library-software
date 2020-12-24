// File: geometry.cpp
// Project: lib
// Created Date: 08/06/2016
// Author: Seki Inoue
// -----
// Last Modified: 22/12/2020
// Modified By: Shun Suzuki (suzuki@hapis.k.u-tokyo.ac.jp)
// -----
// Copyright (c) 2016-2020 Hapis Lab. All rights reserved.
//

#include <stdio.h>

#include <map>

#if WIN32
#include <codeanalysis\warnings.h>
#pragma warning(push)
#pragma warning(disable : ALL_CODE_ANALYSIS_WARNINGS)
#endif
#include <Eigen/Geometry>
#if WIN32
#pragma warning(pop)
#endif

#include "autd3.hpp"
#include "controller.hpp"
#include "privdef.hpp"
#include "vector3.hpp"

using autd::IS_MISSING_TRANSDUCER;
using autd::NUM_TRANS_IN_UNIT;
using autd::NUM_TRANS_X;
using autd::NUM_TRANS_Y;
using autd::TRANS_SIZE_MM;

namespace autd {

class Device {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  static std::unique_ptr<Device> Create(Eigen::Vector3d position, Eigen::Quaterniond quaternion) {
    auto device = std::make_unique<Device>();

    const Eigen::Affine3d transform_matrix = Eigen::Translation3d(position) * quaternion;
    device->x_direction = quaternion * Eigen::Vector3d(1, 0, 0);
    device->y_direction = quaternion * Eigen::Vector3d(0, 1, 0);
    device->z_direction = quaternion * Eigen::Vector3d(0, 0, 1);

    Eigen::Matrix<double, 3, NUM_TRANS_IN_UNIT> local_trans_positions;

    auto index = 0;
    for (size_t y = 0; y < NUM_TRANS_Y; y++)
      for (size_t x = 0; x < NUM_TRANS_X; x++)
        if (!IS_MISSING_TRANSDUCER(x, y)) local_trans_positions.col(index++) = Eigen::Vector3d(x * TRANS_SIZE_MM, y * TRANS_SIZE_MM, 0);

    device->global_trans_positions = transform_matrix * local_trans_positions;

    return device;
  }

  static std::unique_ptr<Device> Create(Eigen::Vector3d position, Eigen::Vector3d euler_angles) {
    const auto quaternion = Eigen::AngleAxis(euler_angles.x(), Eigen::Vector3d::UnitZ()) *
                            Eigen::AngleAxis(euler_angles.y(), Eigen::Vector3d::UnitY()) *
                            Eigen::AngleAxis(euler_angles.z(), Eigen::Vector3d::UnitZ());

    return Create(position, quaternion);
  }

  Eigen::Matrix<double, 3, NUM_TRANS_IN_UNIT> global_trans_positions;
  Eigen::Vector3d x_direction;
  Eigen::Vector3d y_direction;
  Eigen::Vector3d z_direction;
};

class AUTDGeometry : public Geometry {
 public:
  ~AUTDGeometry() override {}

  size_t AddDevice(Vector3 position, Vector3 euler_angles, size_t group = 0) final;
  size_t AddDeviceQuaternion(Vector3 position, Quaternion quaternion, size_t group = 0) final;

  const size_t numDevices() noexcept final;
  const size_t numTransducers() noexcept final;
  size_t GroupIDForDeviceIdx(size_t device_id) final;
  const Vector3 position(size_t transducer_idx) final;
  const Vector3 local_position(size_t device, Vector3 global_position) final;
  const Vector3 direction(size_t transducer_id) final;
  const Vector3 x_direction(size_t transducer_id) final;
  const Vector3 y_direction(size_t transducer_id) final;
  const Vector3 z_direction(size_t transducer_id) final;
  const size_t deviceIdxForTransIdx(size_t transducer_idx) final;

 private:
  std::vector<std::unique_ptr<Device>> _devices;
  std::map<size_t, size_t> _group_map;
};

GeometryPtr Geometry::Create() { return std::make_shared<AUTDGeometry>(); }

size_t AUTDGeometry::AddDevice(Vector3 position, Vector3 euler_angles, size_t group) {
  const size_t device_id = this->_devices.size();
  const auto pos = Eigen::Vector3d(position.x(), position.y(), position.z());
  const auto ea = Eigen::Vector3d(euler_angles.x(), euler_angles.y(), euler_angles.z());
  this->_devices.push_back(Device::Create(pos, ea));
  this->_group_map[device_id] = group;
  return device_id;
}

size_t AUTDGeometry::AddDeviceQuaternion(Vector3 position, Quaternion quaternion, size_t group) {
  const auto device_id = this->_devices.size();
  const auto pos = Eigen::Vector3d(position.x(), position.y(), position.z());
  const auto qua = Eigen::Quaterniond(quaternion.w(), quaternion.x(), quaternion.y(), quaternion.z());
  this->_devices.push_back(Device::Create(pos, qua));
  this->_group_map[device_id] = group;
  return device_id;
}

const size_t AUTDGeometry::numDevices() noexcept { return this->_devices.size(); }

const size_t AUTDGeometry::numTransducers() noexcept { return this->numDevices() * NUM_TRANS_IN_UNIT; }

size_t AUTDGeometry::GroupIDForDeviceIdx(size_t device_idx) { return this->_group_map[device_idx]; }

const Vector3 AUTDGeometry::position(size_t transducer_id) {
  const auto local_trans_id = transducer_id % NUM_TRANS_IN_UNIT;
  const auto pos = this->_devices.at(this->deviceIdxForTransIdx(transducer_id))->global_trans_positions.col(local_trans_id);
  return Vector3(pos.x(), pos.y(), pos.z());
}

const Vector3 AUTDGeometry::local_position(size_t device_idx, Vector3 global_position) {
  const Device* device = this->_devices.at(device_idx).get();
  const auto local_origin = device->global_trans_positions.col(0);
  const Eigen::Vector3d x_dir = device->x_direction;
  const Eigen::Vector3d y_dir = device->y_direction;
  const Eigen::Vector3d z_dir = device->z_direction;
  const auto _global_position = Eigen::Vector3d(global_position.x(), global_position.y(), global_position.z());
  const auto rv = _global_position - local_origin;
  return Vector3(rv.dot(x_dir), rv.dot(y_dir), rv.dot(z_dir));
}

const Vector3 AUTDGeometry::direction(size_t transducer_id) { return z_direction(transducer_id); }

const Vector3 AUTDGeometry::x_direction(size_t transducer_id) {
  const Eigen::Vector3d dir = this->_devices.at(this->deviceIdxForTransIdx(transducer_id))->x_direction;
  return Vector3(dir.x(), dir.y(), dir.z());
}

const Vector3 AUTDGeometry::y_direction(size_t transducer_id) {
  const Eigen::Vector3d dir = this->_devices.at(this->deviceIdxForTransIdx(transducer_id))->y_direction;
  return Vector3(dir.x(), dir.y(), dir.z());
}

const Vector3 AUTDGeometry::z_direction(size_t transducer_id) {
  const Eigen::Vector3d dir = this->_devices.at(this->deviceIdxForTransIdx(transducer_id))->z_direction;
  return Vector3(dir.x(), dir.y(), dir.z());
}

const size_t AUTDGeometry::deviceIdxForTransIdx(size_t transducer_id) { return transducer_id / NUM_TRANS_IN_UNIT; }
}  // namespace autd
