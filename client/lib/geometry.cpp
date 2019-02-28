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
#include "privdef.hpp"

class Device {
public:
	EIGEN_MAKE_ALIGNED_OPERATOR_NEW
		Device(int device_id, Eigen::Vector3f position, Eigen::Vector3f euler_angles)
		: device_id(device_id), position(position), euler_angles(euler_angles) {
		quaternion =
			Eigen::AngleAxisf(euler_angles.x(), Eigen::Vector3f::UnitZ()) *
			Eigen::AngleAxisf(euler_angles.y(), Eigen::Vector3f::UnitY()) *
			Eigen::AngleAxisf(euler_angles.z(), Eigen::Vector3f::UnitZ());


		transform_matrix = Eigen::Translation3f(position) * quaternion;
		z_direction = quaternion * Eigen::Vector3f(0, 0, 1);

		int index = 0;
		for (int y = 0; y < NUM_TRANS_Y; y++)
			for (int x = 0; x < NUM_TRANS_X; x++)
				if (!IS_MISSING_TRANSDUCER(x, y))
					local_trans_positions.col(index++) = Eigen::Vector3f(x*TRANS_SIZE, y*TRANS_SIZE, 0);

		global_trans_positions = transform_matrix * local_trans_positions;
	};
	Device(int device_id, Eigen::Vector3f position, Eigen::Quaternionf quo)
		: device_id(device_id), position(position),quaternion(quo) {

		transform_matrix = Eigen::Translation3f(position) * quo;
		z_direction = quo * Eigen::Vector3f(0, 0, 1);

		euler_angles = quaternion.toRotationMatrix().eulerAngles(2,1,2);

		int index = 0;
		for (int y = 0; y < NUM_TRANS_Y; y++)
			for (int x = 0; x < NUM_TRANS_X; x++)
				if (!IS_MISSING_TRANSDUCER(x, y))
					local_trans_positions.col(index++) = Eigen::Vector3f(x*TRANS_SIZE, y*TRANS_SIZE, 0);

		global_trans_positions = transform_matrix * local_trans_positions;
	};
	int device_id;
	Eigen::Vector3f position;
	Eigen::Matrix<float, 3, NUM_TRANS_IN_UNIT> local_trans_positions;
	Eigen::Matrix<float, 3, NUM_TRANS_IN_UNIT> global_trans_positions;
	Eigen::Vector3f euler_angles;
	Eigen::Quaternionf quaternion;
	Eigen::Vector3f z_direction;
	Eigen::Affine3f transform_matrix;
};

class autd::Geometry::impl {
public:
	std::vector<std::shared_ptr<Device>> devices;
	std::map<int, int> groupMap;
	std::shared_ptr<Device> device(int transducer_id) {
		int eid = transducer_id / NUM_TRANS_IN_UNIT;
		return this->devices[eid];
	}
};

autd::GeometryPtr autd::Geometry::Create() {
	return GeometryPtr(new Geometry);
}

autd::Geometry::Geometry() {
	this->_pimpl = std::unique_ptr<impl>(new impl());
}

autd::Geometry::~Geometry() {
}

int autd::Geometry::AddDevice(Eigen::Vector3f position, Eigen::Vector3f euler_angles, int group) {
	int device_id = (int)(this->_pimpl->devices.size());
	this->_pimpl->devices.push_back(std::shared_ptr<Device>(new Device(device_id, position, euler_angles)));
	this->_pimpl->groupMap[device_id] = group;
	return device_id;
}

int autd::Geometry::AddDeviceQuaternion(Eigen::Vector3f position, Eigen::Quaternionf quaternion, int group) {
	int device_id = (int)(this->_pimpl->devices.size());
	this->_pimpl->devices.push_back(std::shared_ptr<Device>(new Device(device_id, position, quaternion)));
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

const int autd::Geometry::numDevices() {
	return (int) this->_pimpl->devices.size();
}

const int autd::Geometry::numTransducers() {
	return this->numDevices()*NUM_TRANS_IN_UNIT;
}

int autd::Geometry::GroupIDForDeviceID(int deviceID) {
	return this->_pimpl->groupMap[deviceID];
}

const Eigen::Vector3f autd::Geometry::position(int transducer_id) {
	const int local_trans_id = transducer_id % NUM_TRANS_IN_UNIT;
	auto device = this->_pimpl->device(transducer_id);
	return device->global_trans_positions.col(local_trans_id);
}

const Eigen::Vector3f &autd::Geometry::direction(int transducer_id) {
	return this->_pimpl->devices[this->deviceIdForTransIdx(transducer_id)]->z_direction;
}

const int autd::Geometry::deviceIdForDeviceIdx(int device_idx) {
	return this->_pimpl->devices[device_idx]->device_id;
}

const int autd::Geometry::deviceIdForTransIdx(int transducer_id) {
	return this->_pimpl->device(transducer_id)->device_id;
}


float  autd::Geometry::frequency() {
	return FPGA_CLOCK / (640.0f + this->_freq_shift);
}

void   autd::Geometry::SetFrequency(float freq) {
	this->_freq_shift = (int8_t)std::min(std::max(FPGA_CLOCK / freq - 640, (float)std::numeric_limits<int8_t>::min()), (float)std::numeric_limits<int8_t>::max());
}