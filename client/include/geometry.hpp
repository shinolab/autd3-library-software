// File: geometry.hpp
// Project: include
// Created Date: 11/04/2018
// Author: Shun Suzuki
// -----
// Last Modified: 24/12/2020
// Modified By: Shun Suzuki (suzuki@hapis.k.u-tokyo.ac.jp)
// -----
// Copyright (c) 2018-2020 Hapis Lab. All rights reserved.
//

#pragma once

#include <memory>

#include "quaternion.hpp"
#include "vector3.hpp"

namespace autd {
using GeometryPtr = std::shared_ptr<Geometry>;

/**
 * @brief AUTD Geometry
 */
class Geometry {
  friend class _internal::AUTDLogic;

 public:
  virtual ~Geometry() {}
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
   * @brief Number of devices
   */
  virtual const size_t numDevices() noexcept = 0;
  /**
   * @brief Number of transducers
   */
  virtual const size_t numTransducers() noexcept = 0;
  /**
   * @brief Convert device ID into group ID
   */
  virtual size_t GroupIDForDeviceIdx(size_t device_iD) = 0;
  /**
   * @brief Position of a transducer specified by id
   */
  virtual const Vector3 position(size_t transducer_idx) = 0;
  /**
   * @brief Convert a global position to a local position
   */
  virtual const Vector3 local_position(size_t device, Vector3 global_position) = 0;
  /**
   * @brief Normalized direction of a transducer specified by id
   */
  virtual const Vector3 direction(size_t transducer_id) = 0;
  /**
   * @brief Normalized long-axis direction of a device which contains a transducer specified by id
   */
  virtual const Vector3 x_direction(size_t transducer_id) = 0;
  /**
   * @brief Normalized short-axis direction of a device which contains a transducer specified by id
   */
  virtual const Vector3 y_direction(size_t transducer_id) = 0;
  /**
   * @brief Same as the direction()
   */
  virtual const Vector3 z_direction(size_t transducer_id) = 0;
  /**
   * @brief Convert transducer index into device ID
   */
  virtual const size_t deviceIdxForTransIdx(size_t transducer_idx) = 0;

  static GeometryPtr Create();
};
}  // namespace autd
