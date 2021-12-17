// File: geometry.hpp
// Project: core
// Created Date: 14/04/2021
// Author: Shun Suzuki
// -----
// Last Modified: 16/12/2021
// Modified By: Shun Suzuki (suzuki@hapis.k.u-tokyo.ac.jp)
// -----
// Copyright (c) 2021 Hapis Lab. All rights reserved.
//

#pragma once

#if _MSC_VER
#pragma warning(push)
#pragma warning(disable : 4068 6031 6255 6294 26450 26451 26454 26495 26812)
#endif
#if defined(__GNUC__) && !defined(__llvm__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wmaybe-uninitialized"
#endif
#include <Eigen/Dense>
#if _MSC_VER
#pragma warning(pop)
#endif
#if defined(__GNUC__) && !defined(__llvm__)
#pragma GCC diagnostic pop
#endif

#include <vector>

#include "hardware_defined.hpp"

namespace autd {
namespace core {

using Vector3 = Eigen::Matrix<double, 3, 1>;
using Vector4 = Eigen::Matrix<double, 4, 1>;
using Matrix4X4 = Eigen::Matrix<double, 4, 4>;
using Quaternion = Eigen::Quaternion<double>;

/**
 * \brief Transduce contains a position and id of a transducer
 */
struct Transducer final {
  explicit Transducer(const size_t id, const double x, const double y, const double z) : _id(id), _position(x, y, z) {}

  [[nodiscard]] size_t id() const { return _id; }
  [[nodiscard]] const Vector3& position() const { return _position; }

 private:
  size_t _id;
  Vector3 _position;
};

/**
 * \brief Device contains an AUTD device geometry.
 */
struct Device {
  Device(const size_t id, const Vector3& position, const Quaternion& quaternion)
      : _id(id),
        _x_direction(quaternion * Vector3(1, 0, 0)),
        _y_direction(quaternion * Vector3(0, 1, 0)),
        _z_direction(quaternion * Vector3(0, 0, 1)) {
    const Eigen::Transform<double, 3, Eigen::Affine> transform_matrix = Eigen::Translation<double, 3>(position) * quaternion;
    size_t i = id * NUM_TRANS_IN_UNIT;
    for (size_t y = 0; y < NUM_TRANS_Y; y++)
      for (size_t x = 0; x < NUM_TRANS_X; x++) {
        if (is_missing_transducer(x, y)) continue;
        const auto local_pos = Vector4(static_cast<double>(x) * TRANS_SPACING_MM, static_cast<double>(y) * TRANS_SPACING_MM, 0, 1);
        const Vector4 global_pos = transform_matrix * local_pos;
        _transducers.emplace_back(i++, global_pos[0], global_pos[1], global_pos[2]);
      }
    _global_to_local = transform_matrix.inverse();
  }

  Device(const size_t id, const Vector3& position, const Vector3& euler_angles)
      : Device(id, position,
               Eigen::AngleAxis<double>(euler_angles.x(), Vector3::UnitZ()) * Eigen::AngleAxis<double>(euler_angles.y(), Vector3::UnitY()) *
                   Eigen::AngleAxis<double>(euler_angles.z(), Vector3::UnitZ())) {}

  [[nodiscard]] size_t id() const noexcept { return _id; }
  [[nodiscard]] const Vector3& x_direction() const noexcept { return _x_direction; }
  [[nodiscard]] const Vector3& y_direction() const noexcept { return _y_direction; }
  [[nodiscard]] const Vector3& z_direction() const noexcept { return _z_direction; }

  /**
   * @brief Convert a global position to a local position
   */
  [[nodiscard]] Vector3 to_local_position(const Vector3& global_position) const {
    const auto homo = Vector4(global_position[0], global_position[1], global_position[2], 1);
    const Vector4 local_position = _global_to_local * homo;
    return {local_position[0], local_position[1], local_position[2]};
  }

  [[nodiscard]] std::vector<Transducer>::const_iterator begin() const { return _transducers.begin(); }
  [[nodiscard]] std::vector<Transducer>::const_iterator end() const { return _transducers.end(); }

  const Transducer& operator[](const size_t i) const { return _transducers[i]; }

 private:
  size_t _id;
  Vector3 _x_direction;
  Vector3 _y_direction;
  Vector3 _z_direction;
  std::vector<Transducer> _transducers;
  Eigen::Transform<double, 3, Eigen::Affine> _global_to_local;
};

/**
 * @brief Geometry of all devices
 */
class Geometry {
 public:
  Geometry() : _wavelength(8.5), _attenuation(0) {}
  ~Geometry() = default;
  Geometry(const Geometry& v) noexcept = delete;
  Geometry& operator=(const Geometry& obj) = delete;
  Geometry(Geometry&& obj) = default;
  Geometry& operator=(Geometry&& obj) = default;

  /**
   * @brief  Add new device with position and rotation. Note that the transform is done with order: Translate -> Rotate
   * @param position Position of transducer #0, which is the one at the lower-left corner.
   * (The lower-left corner is the one with the two missing transducers.)
   * @param euler_angles ZYZ convention euler angle of the device
   * @return an id of added device
   */
  size_t add_device(const Vector3& position, const Vector3& euler_angles) {
    const auto device_id = this->_devices.size();
    this->_devices.emplace_back(device_id, position, euler_angles);
    return device_id;
  }

  /**
   * @brief Same as add_device(const Vector3&, const Vector3&, const size_t), but using quaternion rather than zyz euler angles.
   * @param position Position of transducer #0, which is the one at the lower-left corner.
   * @param quaternion rotation quaternion of the device.
   * @return an id of added device
   */
  size_t add_device(const Vector3& position, const Quaternion& quaternion) {
    const auto device_id = this->_devices.size();
    this->_devices.emplace_back(device_id, position, quaternion);
    return device_id;
  }

  /**
   * @brief ultrasound wavelength
   */
  double& wavelength() noexcept { return this->_wavelength; }

  /**
   * @brief ultrasound wavelength
   */
  [[nodiscard]] double wavelength() const noexcept { return this->_wavelength; }

  /**
   * @brief attenuation coefficient
   */
  double& attenuation_coefficient() noexcept { return this->_attenuation; }

  /**
   * @brief attenuation coefficient
   */
  [[nodiscard]] double attenuation_coefficient() const noexcept { return this->_attenuation; }

  /**
   * @brief Number of devices
   */
  [[nodiscard]] size_t num_devices() const noexcept { return this->_devices.size(); }

  /**
   * @brief Number of transducers
   */
  [[nodiscard]] size_t num_transducers() const noexcept { return this->num_devices() * NUM_TRANS_IN_UNIT; }

  [[nodiscard]] std::vector<Device>::const_iterator begin() const { return _devices.begin(); }
  [[nodiscard]] std::vector<Device>::const_iterator end() const { return _devices.end(); }

  const Device& operator[](const size_t i) const { return _devices[i]; }

 private:
  std::vector<Device> _devices;
  double _wavelength;
  double _attenuation;
};

}  // namespace core
}  // namespace autd
