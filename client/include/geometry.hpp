// File: geometry.hpp
// Project: include
// Created Date: 11/04/2018
// Author: Shun Suzuki
// -----
// Last Modified: 01/07/2020
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
  friend class AUTDController;

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
  virtual int AddDevice(Vector3 position, Vector3 euler_angles, int group = 0) = 0;
  /**
   * @brief  Add new device with position and rotation. Note that the transform is done with order: Translate -> Rotate
   * @param position Position of transducer #0, which is the one at the lower right corner.
   * (The corner with two lacks of the transducer is the lower left.)
   * @param quaternion rotation quaternion of the device.
   * @param group Grouping ID of the device used in gain::GroupedGain
   * @return an id of added device, which is used to delete or do other device specific controls.
   */
  virtual int AddDeviceQuaternion(Vector3 position, Quaternion quaternion, int group = 0) = 0;
  /**
   * @brief Remove device from the geometry.
   */
  virtual void DelDevice(int device_id) = 0;
  /**
   * @brief Number of devices
   */
  virtual const int numDevices() noexcept = 0;
  /**
   * @brief Number of transducers
   */
  virtual const int numTransducers() noexcept = 0;
  /**
   * @brief Convert device ID into group ID
   */
  virtual int GroupIDForDeviceID(int device_iD) = 0;
  /**
   * @brief Position of a transducer specified by id
   */
  virtual const Vector3 position(int transducer_idx) = 0;
  /**
   * @brief Convert a global position to a local position
   */
  virtual const Vector3 local_position(int device, Vector3 global_position) = 0;
  /**
   * @brief Normalized direction of a transducer specified by id
   */
  virtual const Vector3 direction(int transducer_id) = 0;
  /**
   * @brief Normalized long-axis direction of a device which contains a transducer specified by id
   */
  virtual const Vector3 x_direction(int transducer_id) = 0;
  /**
   * @brief Normalized short-axis direction of a device which contains a transducer specified by id
   */
  virtual const Vector3 y_direction(int transducer_id) = 0;
  /**
   * @brief Same as the direction()
   */
  virtual const Vector3 z_direction(int transducer_id) = 0;
  /**
   * @brief Convert transducer index into device ID
   */
  virtual const int deviceIdForTransIdx(int transducer_idx) = 0;
  /**
   * @brief Convert device index into device ID
   */
  virtual const int deviceIdForDeviceIdx(int device_index) = 0;

 private:
  static GeometryPtr Create();
};
}  // namespace autd
