// File: geometry.cpp
// Project: lib
// Created Date: 08/06/2016
// Author: Seki Inoue
// -----
// Last Modified: 20/02/2020
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

class Device {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  static std::shared_ptr<Device> Create(int device_id, Eigen::Vector3f position, Eigen::Quaternionf quaternion) {
    auto device = std::make_shared<Device>();

    device->device_id = device_id;

    const Eigen::Affine3f transform_matrix = Eigen::Translation3f(position) * quaternion;
    device->z_direction = quaternion * Eigen::Vector3f(0, 0, 1);

    Eigen::Matrix<float, 3, NUM_TRANS_IN_UNIT> local_trans_positions;

    auto index = 0;
    for (int y = 0; y < NUM_TRANS_Y; y++)
      for (int x = 0; x < NUM_TRANS_X; x++)
        if (!IS_MISSING_TRANSDUCER(x, y)) local_trans_positions.col(index++) = Eigen::Vector3f(x * TRANS_SIZE_MM, y * TRANS_SIZE_MM, 0);

    device->global_trans_positions = transform_matrix * local_trans_positions;

    return device;
  }

  static std::shared_ptr<Device> Create(int device_id, Eigen::Vector3f position, Eigen::Vector3f euler_angles) {
    const auto quaternion = Eigen::AngleAxisf(euler_angles.x(), Eigen::Vector3f::UnitZ()) *
                            Eigen::AngleAxisf(euler_angles.y(), Eigen::Vector3f::UnitY()) *
                            Eigen::AngleAxisf(euler_angles.z(), Eigen::Vector3f::UnitZ());

    return Create(device_id, position, quaternion);
  }

  int device_id = 0;
  Eigen::Matrix<float, 3, NUM_TRANS_IN_UNIT> global_trans_positions;
  Eigen::Vector3f z_direction;
};

namespace autd {
class AUTDGeometry : public Geometry {
 public:
  int AddDevice(Eigen::Vector3f position, Eigen::Vector3f euler_angles, int group = 0) final;
  int AddDeviceQuaternion(Eigen::Vector3f position, Eigen::Quaternionf quaternion, int group = 0) final;

  void DelDevice(int device_id) final;
  const int numDevices() noexcept final;
  const int numTransducers() noexcept final;
  int GroupIDForDeviceID(int deviceID) final;
  const Eigen::Vector3f position(int transducer_idx) final;
  const Eigen::Vector3f &direction(int transducer_id) final;
  const int deviceIdForTransIdx(int transducer_idx) final;
  const int deviceIdForDeviceIdx(int device_index) final;

 private:
  std::vector<std::shared_ptr<Device>> devices;
  std::map<int, int> groupMap;
  std::shared_ptr<Device> device(int transducer_id) {
    const auto eid = transducer_id / NUM_TRANS_IN_UNIT;
    return this->devices.at(eid);
  }
};

GeometryPtr Geometry::Create() { return std::make_shared<AUTDGeometry>(); }

int AUTDGeometry::AddDevice(Eigen::Vector3f position, Eigen::Vector3f euler_angles, int group) {
  const auto device_id = static_cast<int>(this->devices.size());
  this->devices.push_back(Device::Create(device_id, position, euler_angles));
  this->groupMap[device_id] = group;
  return device_id;
}

int AUTDGeometry::AddDeviceQuaternion(Eigen::Vector3f position, Eigen::Quaternionf quaternion, int group) {
  const auto device_id = static_cast<int>(this->devices.size());
  this->devices.push_back(Device::Create(device_id, position, quaternion));
  this->groupMap[device_id] = group;
  return device_id;
}

void AUTDGeometry::DelDevice(int device_id) {
  auto itr = this->devices.begin();
  while (itr != this->devices.end()) {
    if ((*itr)->device_id == device_id)
      itr = this->devices.erase(itr);
    else
      itr++;
  }
}

const int AUTDGeometry::numDevices() noexcept { return static_cast<int>(this->devices.size()); }

const int AUTDGeometry::numTransducers() noexcept { return this->numDevices() * NUM_TRANS_IN_UNIT; }

int AUTDGeometry::GroupIDForDeviceID(int deviceID) { return this->groupMap[deviceID]; }

const Eigen::Vector3f AUTDGeometry::position(int transducer_id) {
  const auto local_trans_id = transducer_id % NUM_TRANS_IN_UNIT;
  auto device = this->device(transducer_id);
  return device->global_trans_positions.col(local_trans_id);
}

const Eigen::Vector3f &AUTDGeometry::direction(int transducer_id) { return this->devices.at(this->deviceIdForTransIdx(transducer_id))->z_direction; }

const int AUTDGeometry::deviceIdForDeviceIdx(int device_idx) { return this->devices.at(device_idx)->device_id; }

const int AUTDGeometry::deviceIdForTransIdx(int transducer_id) { return this->device(transducer_id)->device_id; }
}  // namespace autd