//
//  geometry.cpp
//  autd3
//
//  Created by Seki Inoue on 6/8/16.
//
//

#include <stdio.h>
#include <map>
#include <codeanalysis\warnings.h>
#pragma warning( push )
#pragma warning ( disable : ALL_CODE_ANALYSIS_WARNINGS )
#include <Eigen/Geometry>
#pragma warning( pop )
#include "autd3.hpp"
#include "controller.hpp"
#include "privdef.hpp"

using namespace Eigen;

class Device {
public:
	EIGEN_MAKE_ALIGNED_OPERATOR_NEW
		static std::shared_ptr<Device> Create(int device_id, Vector3f position, Quaternionf quaternion) {

		auto device = std::make_shared<Device>();

		device->device_id = device_id;

		const Affine3f transform_matrix = Translation3f(position) * quaternion;
		device->z_direction = quaternion * Vector3f(0, 0, 1);

		Matrix<float, 3, NUM_TRANS_IN_UNIT> local_trans_positions;

		auto index = 0;
		for (int y = 0; y < NUM_TRANS_Y; y++)
			for (int x = 0; x < NUM_TRANS_X; x++)
				if (!IS_MISSING_TRANSDUCER(x, y))
					local_trans_positions.col(index++) = Vector3f(x * TRANS_SIZE_MM, y * TRANS_SIZE_MM, 0);

		device->global_trans_positions = transform_matrix * local_trans_positions;

		return device;
	}

	static std::shared_ptr<Device> Create(int device_id, Vector3f position, Vector3f euler_angles) {
		const auto quaternion =
			AngleAxisf(euler_angles.x(), Vector3f::UnitZ()) *
			AngleAxisf(euler_angles.y(), Vector3f::UnitY()) *
			AngleAxisf(euler_angles.z(), Vector3f::UnitZ());

		return Create(device_id, position, quaternion);
	}

	int device_id;
	Matrix<float, 3, NUM_TRANS_IN_UNIT> global_trans_positions;
	Vector3f z_direction;
};

class autd::Geometry::impl {
public:
	std::vector<std::shared_ptr<Device>> devices;
	std::map<int, int> groupMap;
	std::shared_ptr<Device> device(int transducer_id) {
		const auto eid = transducer_id / NUM_TRANS_IN_UNIT;
		return this->devices.at(eid);
	}
};

autd::GeometryPtr autd::Geometry::Create() {
	return std::make_shared<Geometry>();
}

autd::Geometry::Geometry() {
	this->_pimpl = std::make_unique<impl>();
	this->_freq_shift = -3;
}

int autd::Geometry::AddDevice(Vector3f position, Vector3f euler_angles, int group) {
	const auto device_id = static_cast<int>(this->_pimpl->devices.size());
	this->_pimpl->devices.push_back(Device::Create(device_id, position, euler_angles));
	this->_pimpl->groupMap[device_id] = group;
	return device_id;
}

int autd::Geometry::AddDeviceQuaternion(Vector3f position, Quaternionf quaternion, int group) {
	const auto device_id = static_cast<int>(this->_pimpl->devices.size());
	this->_pimpl->devices.push_back(Device::Create(device_id, position, quaternion));
	this->_pimpl->groupMap[device_id] = group;
	return device_id;
}

void autd::Geometry::DelDevice(int device_id) {
	auto itr = this->_pimpl->devices.begin();
	while (itr != this->_pimpl->devices.end())
	{
		if ((*itr)->device_id == device_id) itr = this->_pimpl->devices.erase(itr);
		else itr++;
	}
}

const int autd::Geometry::numDevices() noexcept {
	return static_cast<int> (this->_pimpl->devices.size());
}

const int autd::Geometry::numTransducers() noexcept {
	return this->numDevices()* NUM_TRANS_IN_UNIT;
}

int autd::Geometry::GroupIDForDeviceID(int deviceID) {
	return this->_pimpl->groupMap[deviceID];
}

const Vector3f autd::Geometry::position(int transducer_id) {
	const auto local_trans_id = transducer_id % NUM_TRANS_IN_UNIT;
	auto device = this->_pimpl->device(transducer_id);
	return device->global_trans_positions.col(local_trans_id);
}

const Vector3f& autd::Geometry::direction(int transducer_id) {
	return this->_pimpl->devices.at(this->deviceIdForTransIdx(transducer_id))->z_direction;
}

const int autd::Geometry::deviceIdForDeviceIdx(int device_idx) {
	return this->_pimpl->devices.at(device_idx)->device_id;
}

const int autd::Geometry::deviceIdForTransIdx(int transducer_id) {
	return this->_pimpl->device(transducer_id)->device_id;
}


float  autd::Geometry::frequency() noexcept {
	return FPGA_CLOCK / (640.0f + this->_freq_shift);
}

void   autd::Geometry::SetFrequency(float freq) noexcept {
	this->_freq_shift = (int8_t)std::min(std::max(FPGA_CLOCK / freq - 640, (float)std::numeric_limits<int8_t>::min()), (float)std::numeric_limits<int8_t>::max());
}