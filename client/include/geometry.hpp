// File: geometry.hpp
// Project: include
// Created Date: 11/04/2018
// Author: Shun Suzuki
// -----
// Last Modified: 27/02/2020
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

class Geometry {
  friend class Controller;

 public:
  static GeometryPtr Create();

  /**
   * @brief  Add new device with position and rotation. Note that the transform is done with order: Translate -> Rotate
   *
   * @param position Position of transducer #0, which is the one at the lower right corner.
   * @param euler_angles ZYZ convention Euler angle of the device.
   *
   * @return an id of added device, which is used to delete or do other device specific controls.
   */
  virtual int AddDevice(Vector3 position, Vector3 euler_angles, int group = 0) = 0;
  virtual int AddDeviceQuaternion(Vector3 position, Quaternion quaternion, int group = 0) = 0;
  /**
   * @brief Remove device from the geometry.
   */
  virtual void DelDevice(int device_id) = 0;
  virtual const int numDevices() noexcept = 0;
  virtual const int numTransducers() noexcept = 0;
  virtual int GroupIDForDeviceID(int device_iD) = 0;
  virtual const Vector3 position(int transducer_idx) = 0;
  /**
   * @brief Normalized direction of a transducer specified by id
   */
  virtual const Vector3 direction(int transducer_id) = 0;
  virtual const int deviceIdForTransIdx(int transducer_idx) = 0;
  virtual const int deviceIdForDeviceIdx(int device_index) = 0;
};
}  // namespace autd
