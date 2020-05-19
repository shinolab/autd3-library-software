// File: geometry.cpp
// Project: lib
// Created Date: 08/06/2016
// Author: Seki Inoue
// -----
// Last Modified: 30/04/2020
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
  static std::shared_ptr<Device> Create(int device_id, Eigen::Vector3d position, Eigen::Quaterniond quaternion) {
    auto device = std::make_shared<Device>();

    device->device_id = device_id;

    const Eigen::Affine3d transform_matrix = Eigen::Translation3d(position) * quaternion;
    device->x_direction = quaternion * Eigen::Vector3d(1, 0, 0);
    device->y_direction = quaternion * Eigen::Vector3d(0, 1, 0);
    device->z_direction = quaternion * Eigen::Vector3d(0, 0, 1);

    Eigen::Matrix<double, 3, NUM_TRANS_IN_UNIT> local_trans_positions;

    auto index = 0;
    for (int y = 0; y < NUM_TRANS_Y; y++)
      for (int x = 0; x < NUM_TRANS_X; x++)
        if (!IS_MISSING_TRANSDUCER(x, y)) local_trans_positions.col(index++) = Eigen::Vector3d(x * TRANS_SIZE_MM, y * TRANS_SIZE_MM, 0);

    device->global_trans_positions = transform_matrix * local_trans_positions;

    return device;
  }

  static std::shared_ptr<Device> Create(int device_id, Eigen::Vector3d position, Eigen::Vector3d euler_angles) {
    const auto quaternion = Eigen::AngleAxis(euler_angles.x(), Eigen::Vector3d::UnitZ()) *
                            Eigen::AngleAxis(euler_angles.y(), Eigen::Vector3d::UnitY()) *
                            Eigen::AngleAxis(euler_angles.z(), Eigen::Vector3d::UnitZ());

    return Create(device_id, position, quaternion);
  }

  int device_id = 0;
  Eigen::Matrix<double, 3, NUM_TRANS_IN_UNIT> global_trans_positions;
  Eigen::Vector3d x_direction;
  Eigen::Vector3d y_direction;
  Eigen::Vector3d z_direction;
};

class AUTDGeometry : public Geometry {
 public:
  ~AUTDGeometry() override {}

  int AddDevice(Vector3 position, Vector3 euler_angles, int group = 0) final;
  int AddDeviceQuaternion(Vector3 position, Quaternion quaternion, int group = 0) final;

  void DelDevice(int device_id) final;
  const int numDevices() noexcept final;
  const int numTransducers() noexcept final;
  int GroupIDForDeviceID(int device_id) final;
  const Vector3 position(int transducer_idx) final;
  const Vector3 direction(int transducer_id) final;
  const Vector3 x_direction(int transducer_id) final;
  const Vector3 y_direction(int transducer_id) final;
  const Vector3 z_direction(int transducer_id) final;
  const int deviceIdForTransIdx(int transducer_idx) final;
  const int deviceIdForDeviceIdx(int device_index) final;

 private:
  std::vector<std::shared_ptr<Device>> _devices;
  std::map<int, int> _group_map;
  std::shared_ptr<Device> device(int transducer_id) {
    const auto eid = transducer_id / NUM_TRANS_IN_UNIT;
    return this->_devices.at(eid);
  }
};

GeometryPtr Geometry::Create() { return std::make_shared<AUTDGeometry>(); }

int AUTDGeometry::AddDevice(Vector3 position, Vector3 euler_angles, int group) {
  const auto device_id = static_cast<int>(this->_devices.size());
  const auto pos = Eigen::Vector3d(position.x(), position.y(), position.z());
  const auto ea = Eigen::Vector3d(euler_angles.x(), euler_angles.y(), euler_angles.z());
  this->_devices.push_back(Device::Create(device_id, pos, ea));
  this->_group_map[device_id] = group;
  return device_id;
}

int AUTDGeometry::AddDeviceQuaternion(Vector3 position, Quaternion quaternion, int group) {
  const auto device_id = static_cast<int>(this->_devices.size());
  const auto pos = Eigen::Vector3d(position.x(), position.y(), position.z());
  const auto qua = Eigen::Quaterniond(quaternion.w(), quaternion.x(), quaternion.y(), quaternion.z());
  this->_devices.push_back(Device::Create(device_id, pos, qua));
  this->_group_map[device_id] = group;
  return device_id;
}

void AUTDGeometry::DelDevice(int device_id) {
  auto itr = this->_devices.begin();
  while (itr != this->_devices.end()) {
    if ((*itr)->device_id == device_id)
      itr = this->_devices.erase(itr);
    else
      itr++;
  }
}

const int AUTDGeometry::numDevices() noexcept { return static_cast<int>(this->_devices.size()); }

const int AUTDGeometry::numTransducers() noexcept { return this->numDevices() * NUM_TRANS_IN_UNIT; }

int AUTDGeometry::GroupIDForDeviceID(int device_id) { return this->_group_map[device_id]; }

const Vector3 AUTDGeometry::position(int transducer_id) {
  const auto local_trans_id = transducer_id % NUM_TRANS_IN_UNIT;
  auto device = this->device(transducer_id);
  const auto pos = device->global_trans_positions.col(local_trans_id);
  return Vector3(pos.x(), pos.y(), pos.z());
}

const Vector3 AUTDGeometry::direction(int transducer_id) { return z_direction(transducer_id); }

const Vector3 AUTDGeometry::x_direction(int transducer_id) {
  const auto dir = this->_devices.at(this->deviceIdForTransIdx(transducer_id))->x_direction;
  return Vector3(dir.x(), dir.y(), dir.z());
}

const Vector3 AUTDGeometry::y_direction(int transducer_id) {
  const auto dir = this->_devices.at(this->deviceIdForTransIdx(transducer_id))->y_direction;
  return Vector3(dir.x(), dir.y(), dir.z());
}

const Vector3 AUTDGeometry::z_direction(int transducer_id) {
  const auto dir = this->_devices.at(this->deviceIdForTransIdx(transducer_id))->z_direction;
  return Vector3(dir.x(), dir.y(), dir.z());
}

const int AUTDGeometry::deviceIdForDeviceIdx(int device_idx) { return this->_devices.at(device_idx)->device_id; }

const int AUTDGeometry::deviceIdForTransIdx(int transducer_id) { return this->device(transducer_id)->device_id; }
}  // namespace autd
