// File: geometry.hpp
// Project: include
// Created Date: 11/04/2018
// Author: Shun Suzuki
// -----
// Last Modified: 04/04/2021
// Modified By: Shun Suzuki (suzuki@hapis.k.u-tokyo.ac.jp)
// -----
// Copyright (c) 2018-2020 Hapis Lab. All rights reserved.
//

#pragma once

#include <memory>

#include "autd_types.hpp"
#include "linalg.hpp"

namespace autd {

class Geometry;
using GeometryPtr = std::shared_ptr<Geometry>;

/**
 * @brief AUTD Geometry
 */
class Geometry {
 public:
  Geometry() = default;
  virtual ~Geometry() = default;
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
  virtual size_t AddDevice(Vector3 position, Vector3 euler_angles, size_t group = 0) = 0;
  /**
   * @brief  Add new device with position and rotation. Note that the transform is done with order: Translate -> Rotate
   * @param position Position of transducer #0, which is the one at the lower right corner.
   * (The corner with two lacks of the transducer is the lower left.)
   * @param quaternion rotation quaternion of the device.
   * @param group Grouping ID of the device used in gain::GroupedGain
   * @return an id of added device, which is used to delete or do other device specific controls.
   */
  virtual size_t AddDeviceQuaternion(Vector3 position, Quaternion quaternion, size_t group = 0) = 0;

  /**
   * @brief Delete device
   * @param idx Index of the device to delete.
   * @return an index of deleted device
   */
  virtual size_t DelDevice(size_t idx) = 0;

  /**
   * @brief Clear all devices
   */
  virtual void ClearDevices() = 0;

  /**
   * @brief ultrasound wavelength
   */
  virtual Float wavelength() noexcept = 0;
  /**
   * @brief set ultrasound wavelength
   */
  virtual void set_wavelength(Float wavelength) noexcept = 0;
  /**
   * @brief attenuation coefficient
   */
  virtual Float attenuation_coeff() noexcept = 0;
  /**
   * @brief set attenuation coefficient
   */
  virtual void set_attenuation_coeff(Float attenuation_coeff) noexcept = 0;
  /**
   * @brief Number of devices
   */
  virtual size_t num_devices() noexcept = 0;
  /**
   * @brief Number of transducers
   */
  virtual size_t num_transducers() noexcept = 0;
  /**
   * @brief Convert device ID into group ID
   */
  virtual size_t group_id_for_device_idx(size_t device_idx) = 0;
  /**
   * @brief Position of a transducer specified by id
   */
  virtual Vector3 position(size_t global_transducer_idx) = 0;
  /**
   * @brief Position of a transducer specified by id
   */
  virtual Vector3 position(size_t device_idx, size_t local_transducer_idx) = 0;
  /**
   * @brief Convert a global position to a local position
   */
  virtual Vector3 local_position(size_t device_idx, Vector3 global_position) = 0;
  /**
   * @brief Normalized direction of a device
   */
  virtual Vector3 direction(size_t device_idx) = 0;
  /**
   * @brief Normalized long-axis direction of a device
   */
  virtual Vector3 x_direction(size_t device_idx) = 0;
  /**
   * @brief Normalized short-axis direction of a device
   */
  virtual Vector3 y_direction(size_t device_idx) = 0;
  /**
   * @brief Same as the direction()
   */
  virtual Vector3 z_direction(size_t device_idx) = 0;
  /**
   * @brief Convert transducer index into device ID
   */
  virtual size_t device_idx_for_trans_idx(size_t transducer_idx) = 0;

  static GeometryPtr Create();
};
}  // namespace autd
