#pragma once

#include <memory>
#include <codeanalysis\warnings.h>
#pragma warning( push )
#pragma warning ( disable : ALL_CODE_ANALYSIS_WARNINGS )
#include <Eigen/Core>
#include <Eigen/Geometry> 
#pragma warning( pop )

namespace autd {
	class Geometry;
	typedef std::shared_ptr<Geometry> GeometryPtr;

	class Geometry
	{
		friend class Controller;
	public:
		Geometry();
		~Geometry();
		static GeometryPtr Create();
		/*!
			@brief Add new device with position and rotation. Note that the transform is done with order: Translate -> Rotate
			@param position Position of transducer #0, which is the one at the lower right corner.
			@param euler_angles ZYZ convention Euler angle of the device.
			@return an id of added device, which is used to delete or do other device specific controls.
		 */
		int AddDevice(Eigen::Vector3f position, Eigen::Vector3f euler_angles, int group = 0);
		int AddDeviceQuaternion(Eigen::Vector3f position, Eigen::Quaternionf quaternion, int group = 0);
		/*!
			@brief Remove device from the geometry.
		 */
		void DelDevice(int device_id);
		const int numDevices();
		const int numTransducers();
		int GroupIDForDeviceID(int deviceID);
		const Eigen::Vector3f position(int transducer_idx);
		/*!
			@brief Normalized direction of a transducer specified by id
		 */
		const Eigen::Vector3f &direction(int transducer_id);
		const int deviceIdForTransIdx(int transducer_idx);
		const int deviceIdForDeviceIdx(int device_index);

		/*!
		@brief Return a frquency of ultrasound.
		*/
		float frequency();
		/*!
		@brief Set a frquency of ultrasound, which should be [33.4kHz < freq < 50.0kHz].
		*/
		void SetFrequency(float freq);
	private:
		int8_t _freq_shift;
		class impl;
		std::unique_ptr<impl> _pimpl;
	};
}
