// File: geometry.cpp
// Project: lib
// Created Date: 08/06/2016
// Author: Seki Inoue
// -----
// Last Modified: 27/02/2021
// Modified By: Shun Suzuki (suzuki@hapis.k.u-tokyo.ac.jp)
// -----
// Copyright (c) 2016-2020 Hapis Lab. All rights reserved.
//

#include "geometry.hpp"

#include <map>

#include "autd3.hpp"
#include "autd_logic.hpp"
#include "linalg.hpp"

using autd::IsMissingTransducer;
using autd::NUM_TRANS_IN_UNIT;
using autd::NUM_TRANS_X;
using autd::NUM_TRANS_Y;
using autd::TRANS_SIZE_MM;

namespace autd {
struct Device {
  static Device Create(const Vector3& position, const Quaternion& quaternion) {
    const Matrix4x4 transform_matrix = Translation(position, quaternion);
    const auto x_direction = quaternion * Vector3(1, 0, 0);
    const auto y_direction = quaternion * Vector3(0, 1, 0);
    const auto z_direction = quaternion * Vector3(0, 0, 1);

    std::unique_ptr<Vector3[]> global_trans_positions = std::make_unique<Vector3[]>(NUM_TRANS_IN_UNIT);

    auto index = 0;
    for (size_t y = 0; y < NUM_TRANS_Y; y++)
      for (size_t x = 0; x < NUM_TRANS_X; x++)
        if (!IsMissingTransducer(x, y)) {
          const auto local_pos = Vector4(static_cast<Float>(x) * TRANS_SIZE_MM, static_cast<Float>(y) * TRANS_SIZE_MM, 0, 1);
          const auto global_pos = transform_matrix * local_pos;
          global_trans_positions[index++] = ToVector3(global_pos);
        }

    return Device{x_direction, y_direction, z_direction, std::move(global_trans_positions)};
  }

  static Device Create(const Vector3& position, const Vector3& euler_angles) {
    const auto quaternion =
        AngleAxis(euler_angles.x(), Vector3::UnitZ()) * AngleAxis(euler_angles.y(), Vector3::UnitY()) * AngleAxis(euler_angles.z(), Vector3::UnitZ());

    return Create(position, quaternion);
  }

  Vector3 x_direction;
  Vector3 y_direction;
  Vector3 z_direction;
  std::unique_ptr<Vector3[]> global_trans_positions;
};

class AUTDGeometry final : public Geometry {
 public:
  AUTDGeometry() : _wavelength(8.5) {}
  ~AUTDGeometry() override = default;
  AUTDGeometry(const AUTDGeometry& v) noexcept = default;
  AUTDGeometry& operator=(const AUTDGeometry& obj) = default;
  AUTDGeometry(AUTDGeometry&& obj) = default;
  AUTDGeometry& operator=(AUTDGeometry&& obj) = default;

  size_t AddDevice(Vector3 position, Vector3 euler_angles, size_t group = 0) override;
  size_t AddDeviceQuaternion(Vector3 position, Quaternion quaternion, size_t group = 0) override;

  Float wavelength() noexcept override;
  void set_wavelength(Float wavelength) noexcept override;

  size_t num_devices() noexcept override;
  size_t num_transducers() noexcept override;
  size_t group_id_for_device_idx(size_t device_idx) override;
  Vector3 position(size_t global_transducer_idx) override;
  Vector3 position(size_t device, size_t local_transducer_idx) override;
  Vector3 local_position(size_t device_idx, Vector3 global_position) override;

  Vector3 direction(size_t device_idx) override;
  Vector3 x_direction(size_t device_idx) override;
  Vector3 y_direction(size_t device_idx) override;
  Vector3 z_direction(size_t device_idx) override;
  size_t device_idx_for_trans_idx(size_t transducer_idx) override;

 private:
  std::vector<Device> _devices;
  std::map<size_t, size_t> _group_map;
  Float _wavelength;
};

GeometryPtr Geometry::Create() { return std::make_shared<AUTDGeometry>(); }

size_t AUTDGeometry::AddDevice(const Vector3 position, const Vector3 euler_angles, const size_t group) {
  const auto device_id = this->_devices.size();
  this->_devices.emplace_back(Device::Create(position, euler_angles));
  this->_group_map[device_id] = group;
  return device_id;
}

size_t AUTDGeometry::AddDeviceQuaternion(const Vector3 position, const Quaternion quaternion, const size_t group) {
  const auto device_id = this->_devices.size();
  this->_devices.emplace_back(Device::Create(position, quaternion));
  this->_group_map[device_id] = group;
  return device_id;
}

Float AUTDGeometry::wavelength() noexcept { return this->_wavelength; }
void AUTDGeometry::set_wavelength(const Float wavelength) noexcept { this->_wavelength = wavelength; }

size_t AUTDGeometry::num_devices() noexcept { return this->_devices.size(); }

size_t AUTDGeometry::num_transducers() noexcept { return this->num_devices() * NUM_TRANS_IN_UNIT; }

size_t AUTDGeometry::group_id_for_device_idx(const size_t device_idx) { return this->_group_map[device_idx]; }

Vector3 AUTDGeometry::position(const size_t global_transducer_idx) {
  const auto local_trans_id = global_transducer_idx % NUM_TRANS_IN_UNIT;
  return position(this->device_idx_for_trans_idx(global_transducer_idx), local_trans_id);
}

Vector3 AUTDGeometry::position(const size_t device, const size_t local_transducer_idx) {
  const auto& dev = this->_devices[device];
  return dev.global_trans_positions[local_transducer_idx];
}

Vector3 AUTDGeometry::local_position(const size_t device_idx, const Vector3 global_position) {
  const auto& device = this->_devices[device_idx];
  const auto& local_origin = device.global_trans_positions[0];
  const auto& x_dir = device.x_direction;
  const auto& y_dir = device.y_direction;
  const auto& z_dir = device.z_direction;
  const auto rv = global_position - local_origin;
  return Vector3(rv.dot(x_dir), rv.dot(y_dir), rv.dot(z_dir));
}

Vector3 AUTDGeometry::direction(const size_t device_idx) { return z_direction(device_idx); }

Vector3 AUTDGeometry::x_direction(const size_t device_idx) {
  const auto& dir = this->_devices[device_idx].x_direction;
  return dir;
}

Vector3 AUTDGeometry::y_direction(const size_t device_idx) {
  const auto& dir = this->_devices[device_idx].x_direction;
  return dir;
}

Vector3 AUTDGeometry::z_direction(const size_t device_idx) {
  const auto& dir = this->_devices[device_idx].x_direction;
  return dir;
}

size_t AUTDGeometry::device_idx_for_trans_idx(const size_t transducer_idx) { return transducer_idx / NUM_TRANS_IN_UNIT; }
}  // namespace autd
