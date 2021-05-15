// File: geometry.hpp
// Project: core
// Created Date: 14/04/2021
// Author: Shun Suzuki
// -----
// Last Modified: 11/05/2021
// Modified By: Shun Suzuki (suzuki@hapis.k.u-tokyo.ac.jp)
// -----
// Copyright (c) 2021 Hapis Lab. All rights reserved.
//

#pragma once

#if WIN32
#pragma warning(push)
#pragma warning(disable : 26450 26495 26812)
#endif
#ifdef linux
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wmaybe-uninitialized"
#endif
#include <Eigen/Dense>
#if WIN32
#pragma warning(pop)
#endif
#ifdef linux
#pragma GCC diagnostic pop
#endif

#include <map>
#include <memory>
#include <utility>
#include <vector>

#include "hardware_defined.hpp"

namespace autd::core {

class Geometry;
using GeometryPtr = std::shared_ptr<Geometry>;

using Vector3 = Eigen::Matrix<double, 3, 1>;
using Vector4 = Eigen::Matrix<double, 4, 1>;
using Matrix4X4 = Eigen::Matrix<double, 4, 4>;
using Quaternion = Eigen::Quaternion<double>;

struct Device {
  static Device Create(const Vector3& position, const Quaternion& quaternion) {
    const Eigen::Transform<double, 3, Eigen::Affine> transform_matrix = Eigen::Translation<double, 3>(position) * quaternion;
    const auto x_direction = quaternion * Vector3(1, 0, 0);
    const auto y_direction = quaternion * Vector3(0, 1, 0);
    const auto z_direction = quaternion * Vector3(0, 0, 1);

    auto global_trans_positions = std::make_unique<Vector3[]>(NUM_TRANS_IN_UNIT);

    auto index = 0;
    for (size_t y = 0; y < NUM_TRANS_Y; y++)
      for (size_t x = 0; x < NUM_TRANS_X; x++)
        if (!IsMissingTransducer(x, y)) {
          const auto local_pos = Vector4(static_cast<double>(x) * TRANS_SPACING_MM, static_cast<double>(y) * TRANS_SPACING_MM, 0, 1);
          const auto global_pos = transform_matrix * local_pos;
          global_trans_positions[index++] = Vector3(global_pos[0], global_pos[1], global_pos[2]);
        }

    return Device{x_direction, y_direction, z_direction, std::move(global_trans_positions)};
  }

  static Device Create(const Vector3& position, const Vector3& euler_angles) {
    const auto quaternion = Eigen::AngleAxis(euler_angles.x(), Vector3::UnitZ()) * Eigen::AngleAxis(euler_angles.y(), Vector3::UnitY()) *
                            Eigen::AngleAxis(euler_angles.z(), Vector3::UnitZ());

    return Create(position, quaternion);
  }

  Vector3 x_direction;
  Vector3 y_direction;
  Vector3 z_direction;
  std::unique_ptr<Vector3[]> global_trans_positions;
};

/**
 * @brief AUTD Geometry
 */
class Geometry {
 public:
  Geometry() : _wavelength(8.5), _attenuation(0) {}
  ~Geometry() = default;
  Geometry(const Geometry& v) noexcept = default;
  Geometry& operator=(const Geometry& obj) = default;
  Geometry(Geometry&& obj) = default;
  Geometry& operator=(Geometry&& obj) = default;

  /**
   * @brief  Add new device with position and rotation. Note that the transform is done with order: Translate -> Rotate
   * @param position Position of transducer #0, which is the one at the lower right corner.
   * (The corner with two lacks of the transducer is the lower left.)
   * @param euler_angles ZYZ convention Euler angle of the device.
   * @param group Grouping ID of the device used in gain::GroupedGain
   * @return an id of added device, which is used to delete or do other device specific controls.
   */
  size_t AddDevice(Vector3 position, Vector3 euler_angles, size_t group = 0) {
    const auto device_id = this->_devices.size();
    this->_devices.emplace_back(Device::Create(position, euler_angles));
    this->_group_map[device_id] = group;
    return device_id;
  }

  /**
   * @brief  Add new device with position and rotation. Note that the transform is done with order: Translate -> Rotate
   * @param position Position of transducer #0, which is the one at the lower right corner.
   * (The corner with two lacks of the transducer is the lower left.)
   * @param quaternion rotation quaternion of the device.
   * @param group Grouping ID of the device used in gain::GroupedGain
   * @return an id of added device, which is used to delete or do other device specific controls.
   */
  size_t AddDevice(Vector3 position, Quaternion quaternion, size_t group = 0) {
    const auto device_id = this->_devices.size();
    this->_devices.emplace_back(Device::Create(position, quaternion));
    this->_group_map[device_id] = group;
    return device_id;
  }

  /**
   * @brief Delete device
   * @param idx Index of the device to delete.
   * @return an index of deleted device
   */
  size_t DelDevice(size_t idx) {
    this->_devices.erase(this->_devices.begin() + idx);
    return idx;
  }

  /**
   * @brief Clear all devices
   */
  void ClearDevices() { std::vector<Device>().swap(this->_devices); }

  /**
   * @brief ultrasound wavelength
   */
  double& wavelength() noexcept { return this->_wavelength; }

  /**
   * @brief attenuation coefficient
   */
  double& attenuation_coeff() noexcept { return this->_attenuation; }

  /**
   * @brief Number of devices
   */
  size_t num_devices() noexcept { return this->_devices.size(); }

  /**
   * @brief Number of transducers
   */
  size_t num_transducers() noexcept { return this->num_devices() * NUM_TRANS_IN_UNIT; }

  /**
   * @brief Convert device ID into group ID
   */
  size_t group_id_for_device_idx(size_t device_idx) { return this->_group_map[device_idx]; }

  /**
   * @brief Position of a transducer specified by id
   */
  Vector3 position(size_t global_transducer_idx) {
    const auto local_trans_id = global_transducer_idx % NUM_TRANS_IN_UNIT;
    return position(this->device_idx_for_trans_idx(global_transducer_idx), local_trans_id);
  }

  /**
   * @brief Position of a transducer specified by id
   */
  Vector3 position(size_t device_idx, size_t local_transducer_idx) {
    const auto& [_x, _y, _z, global_trans_positions] = this->_devices[device_idx];
    return global_trans_positions[local_transducer_idx];
  }

  /**
   * @brief Convert a global position to a local position
   */
  Vector3 local_position(size_t device_idx, Vector3 global_position) {
    const auto& [x_direction, y_direction, z_direction, global_trans_positions] = this->_devices[device_idx];
    const auto& local_origin = global_trans_positions[0];
    const auto& x_dir = x_direction;
    const auto& y_dir = y_direction;
    const auto& z_dir = z_direction;
    const auto rv = global_position - local_origin;
    return Vector3(rv.dot(x_dir), rv.dot(y_dir), rv.dot(z_dir));
  }

  /**
   * @brief Normalized direction of a device
   */
  Vector3 direction(size_t device_idx) { return this->_devices[device_idx].z_direction; }
  /**
   * @brief Normalized long-axis direction of a device
   */
  Vector3 x_direction(size_t device_idx) { return this->_devices[device_idx].x_direction; }

  /**
   * @brief Normalized short-axis direction of a device
   */
  Vector3 y_direction(size_t device_idx) { return this->_devices[device_idx].y_direction; }

  /**
   * @brief Same as the direction()
   */
  Vector3 z_direction(size_t device_idx) { return this->_devices[device_idx].z_direction; }

  /**
   * @brief Convert transducer index into device ID
   */
  size_t device_idx_for_trans_idx(size_t transducer_idx) { return transducer_idx / NUM_TRANS_IN_UNIT; }

 private:
  std::vector<Device> _devices;
  std::map<size_t, size_t> _group_map;
  double _wavelength;
  double _attenuation;
};
}  // namespace autd::core
