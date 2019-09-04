/*
 * File: geometry.hpp
 * Project: include
 * Created Date: 11/04/2018
 * Author: Shun Suzuki
 * -----
 * Last Modified: 04/09/2019
 * Modified By: Shun Suzuki (suzuki@hapis.k.u-tokyo.ac.jp)
 * -----
 * Copyright (c) 2019 Hapis Lab. All rights reserved.
 * 
 */

#pragma once

#include <memory>
#if WIN32
#include <codeanalysis\warnings.h>
#pragma warning(push)
#pragma warning(disable \
				: ALL_CODE_ANALYSIS_WARNINGS)
#endif
#include <Eigen/Core>
#include <Eigen/Geometry>
#if WIN32
#pragma warning(pop)
#endif

using namespace Eigen;

namespace autd
{
class Geometry;
using GeometryPtr = std::shared_ptr<Geometry>;

class Geometry
{
	friend class Controller;

public:
	Geometry();
	static GeometryPtr Create();
	/*!
			@brief Add new device with position and rotation. Note that the transform is done with order: Translate -> Rotate
			@param position Position of transducer #0, which is the one at the lower right corner.
			@param euler_angles ZYZ convention Euler angle of the device.
			@return an id of added device, which is used to delete or do other device specific controls.
		 */
	int AddDevice(Vector3f position, Vector3f euler_angles, int group = 0);
	int AddDeviceQuaternion(Vector3f position, Quaternionf quaternion, int group = 0);
	/*!
			@brief Remove device from the geometry.
		 */
	void DelDevice(int device_id);
	const int numDevices() noexcept;
	const int numTransducers() noexcept;
	int GroupIDForDeviceID(int deviceID);
	const Vector3f position(int transducer_idx);
	/*!
			@brief Normalized direction of a transducer specified by id
		 */
	const Vector3f &direction(int transducer_id);
	const int deviceIdForTransIdx(int transducer_idx);
	const int deviceIdForDeviceIdx(int device_index);

	/*!
		@brief Return a frquency of ultrasound.
		*/
	float frequency() noexcept;
	/*!
		@brief Set a frquency of ultrasound, which should be [33.4kHz < freq < 50.0kHz].
		*/
	void SetFrequency(float freq) noexcept;

private:
	int8_t _freq_shift;
	class impl;
	std::unique_ptr<impl> _pimpl;
};
} // namespace autd
