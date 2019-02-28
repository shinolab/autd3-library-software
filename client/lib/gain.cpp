//
//  gain.cpp
//  autd3
//
//  Created by Seki Inoue on 6/1/16.
//  Changed by Shun Suzuki on 02/07/2018.
//

#include <iostream>
#include <string>
#include <map>
#include <vector>
#include <codeanalysis\warnings.h>
#pragma warning( push )
#pragma warning( disable : ALL_CODE_ANALYSIS_WARNINGS )
#include <boost/assert.hpp>
#pragma warning( pop )
#include "privdef.hpp"
#include "autd3.hpp"

autd::GainPtr autd::Gain::Create() {
	return CreateHelper<Gain>();
}

autd::Gain::Gain() {
	this->_built = false;
	this->_fix = false;
	this->_geometry = GeometryPtr(nullptr);
}

void autd::Gain::build() {
	if (this->built()) return;
	if (this->geometry().get() == nullptr) BOOST_ASSERT_MSG(false, "Geometry is required to build Gain");

	for (int i = 0; i < this->geometry()->numDevices(); i++) {
		this->_data[this->geometry()->deviceIdForDeviceIdx(i)] = std::vector<uint16_t>(NUM_TRANS_IN_UNIT, 0x0000);
	}
}

bool autd::Gain::built() {
	return this->_built;
}

void autd::Gain::SetGeometry(const autd::GeometryPtr &geometry) {
	this->_geometry = geometry;
}

inline void ConvertAmpPhase(uint8_t amp_i, uint8_t phase_i, uint8_t& amp_o, uint8_t& phase_o)
{
	double d = asin(amp_i / 255.0) / M_PIf;  // duty (0 ~ 0.5)
	amp_o = static_cast<uint8_t>(511 * d);
	phase_o = static_cast<uint8_t>((int)(phase_i + 256 - 128 * d) % 256);
}

void autd::Gain::Fix() {
	this->_fix = true;
}

void autd::Gain::FixImpl() {
	for (int i = 0; i < this->geometry()->numDevices(); i++) {
		std::vector<uint16_t> *vec = &this->_data[this->geometry()->deviceIdForDeviceIdx(i)];
		for (size_t j = 0; j < vec->size(); j++)
		{
			auto amp = (uint8_t)((*vec)[j] >> 8);
			auto phase = (uint8_t)(*vec)[j];
			uint8_t amp_fix, phase_fix;
			ConvertAmpPhase(amp, phase, amp_fix, phase_fix);
			(*vec)[j] = (uint16_t)(amp_fix << 8) + phase_fix;
		}
	}
}

autd::GeometryPtr autd::Gain::geometry() {
	return this->_geometry;
}

std::map<int, std::vector<uint16_t> > autd::Gain::data() {
	return this->_data;
}

autd::GainPtr autd::PlaneWaveGain::Create(Eigen::Vector3f direction) {
	auto ptr = CreateHelper<PlaneWaveGain>();
	ptr->_direction = direction;
	return ptr;
}

void autd::PlaneWaveGain::build() {
	if (this->built()) return;
	if (this->geometry() == nullptr) BOOST_ASSERT_MSG(false, "Geometry is required to build Gain");

	this->_data.clear();
	const int ndevice = this->geometry()->numDevices();
	for (int i = 0; i < ndevice; i++) {
		this->_data[this->geometry()->deviceIdForDeviceIdx(i)].resize(NUM_TRANS_IN_UNIT);
	}

	const int ntrans = this->geometry()->numTransducers();
	for (int i = 0; i < ntrans; i++) {
		this->_data[this->geometry()->deviceIdForTransIdx(i)][i%NUM_TRANS_IN_UNIT] = 0xFF00;
	}

	this->_built = true;
}

autd::GainPtr autd::FocalPointGain::Create(Eigen::Vector3f point) {
	return autd::FocalPointGain::Create(point, 255);
}

autd::GainPtr autd::FocalPointGain::Create(Eigen::Vector3f point, uint8_t amp) {
	auto gain = CreateHelper<FocalPointGain>();
	gain->_point = point;
	gain->_geometry = GeometryPtr(nullptr);
	gain->_amp = amp;
	return gain;
}

void autd::FocalPointGain::build() {
	if (this->built()) return;
	if (this->geometry() == nullptr) BOOST_ASSERT_MSG(false, "Geometry is required to build Gain");

	this->_data.clear();

	const int ndevice = this->geometry()->numDevices();
	for (int i = 0; i < ndevice; i++) {
		this->_data[this->geometry()->deviceIdForDeviceIdx(i)].resize(NUM_TRANS_IN_UNIT);
	}

	const int ntrans = this->geometry()->numTransducers();
	for (int i = 0; i < ntrans; i++) {
		Eigen::Vector3f trp = this->geometry()->position(i);
		float dist = (trp - this->_point).norm();
		float fphase = fmodf(dist, ULTRASOUND_WAVELENGTH) / ULTRASOUND_WAVELENGTH;
		uint8_t amp = this->_amp;
		uint8_t phase = (uint8_t)round(255.0*(1 - fphase));
		this->_data[this->geometry()->deviceIdForTransIdx(i)][i%NUM_TRANS_IN_UNIT] = ((uint16_t)amp << 8) + phase;
	}

	this->_built = true;
}


autd::GainPtr autd::BesselBeamGain::Create(Eigen::Vector3f point, Eigen::Vector3f vec_n, float theta_z) {
	auto gain = CreateHelper<BesselBeamGain>();
	gain->_point = point;
	gain->_vec_n = vec_n;
	gain->_theta_z = theta_z;
	gain->_geometry = GeometryPtr(nullptr);
	return gain;
}

void autd::BesselBeamGain::build() {
	if (this->built()) return;
	if (this->geometry() == nullptr) BOOST_ASSERT_MSG(false, "Geometry is required to build Gain");

	this->_data.clear();
	const int ndevice = this->geometry()->numDevices();
	for (int i = 0; i < ndevice; i++) {
		this->_data[this->geometry()->deviceIdForDeviceIdx(i)].resize(NUM_TRANS_IN_UNIT);
	}
	const int ntrans = this->geometry()->numTransducers();

	Eigen::Vector3f _ez;
	_ez << 0, 0, 1.0f;

	Eigen::Vector3f _v;

	if (_vec_n.norm() > 0) _vec_n = _vec_n / _vec_n.norm();
	_v << _vec_n.y(), -_vec_n.x(), 0;
	float _theta_w = asinf(_v.norm());

	for (int i = 0; i < ntrans; i++) {
		Eigen::Vector3f trp = this->geometry()->position(i);
		Eigen::Vector3f _r = trp - this->_point;
		Eigen::Vector3f _v_x_r;
		_v_x_r << _v.y() * _r.z() - _v.z() * _r.y(), _v.z() * _r.x() - _v.x() * _r.z(), _v.x() * _r.y() - _v.y() * _r.x();
		Eigen::Vector3f _R = cos(_theta_w) * _r + sin(_theta_w) * _v_x_r + _v.dot(_r) * (1.0f - cos(_theta_w)) * _v;
		float fphase = fmodf(sin(_theta_z) * sqrt(_R.x() * _R.x() + _R.y() * _R.y()) - cos(_theta_z) * _R.z(), ULTRASOUND_WAVELENGTH) / ULTRASOUND_WAVELENGTH;
		uint8_t amp = 0xff;
		uint8_t phase = (uint8_t)round(255.0*(1 - fphase));
		this->_data[this->geometry()->deviceIdForTransIdx(i)][i%NUM_TRANS_IN_UNIT] = ((uint16_t)amp << 8) + phase;
	}

	this->_built = true;
}

autd::GainPtr autd::CustomGain::Create(uint16_t* data, int dataLength) {
	auto gain = CreateHelper<CustomGain>();;
	gain->_rawdata.resize(dataLength);
	for (int i = 0; i < dataLength; i++) gain->_rawdata[i] = data[i];
	gain->_geometry = GeometryPtr(nullptr);
	return gain;
}

void autd::CustomGain::build() {
	if (this->built()) return;
	if (this->geometry() == nullptr) BOOST_ASSERT_MSG(false, "Geometry is required to build Gain");

	this->_data.clear();
	const int ndevice = this->geometry()->numDevices();
	for (int i = 0; i < ndevice; i++) {
		this->_data[this->geometry()->deviceIdForDeviceIdx(i)].resize(NUM_TRANS_IN_UNIT);
	}
	const int ntrans = this->geometry()->numTransducers();

	for (int i = 0; i < ntrans; i++) {
		this->_data[this->geometry()->deviceIdForTransIdx(i)][i%NUM_TRANS_IN_UNIT] = this->_rawdata[i];
	}

	this->_built = true;
}

autd::GainPtr autd::TransducerTestGain::Create(int idx, int amp, int phase) {
	auto gain = CreateHelper<TransducerTestGain>();
	gain->_xdcr_idx = idx;
	gain->_amp = amp;
	gain->_phase = phase;
	return gain;
}

void autd::TransducerTestGain::build() {
	if (this->built()) return;
	if (this->geometry() == nullptr) BOOST_ASSERT_MSG(false, "Geometry is required to build Gain");

	this->_data.clear();
	const int ndevice = this->geometry()->numDevices();
	for (int i = 0; i < ndevice; i++) {
		this->_data[this->geometry()->deviceIdForDeviceIdx(i)].resize(NUM_TRANS_IN_UNIT);
	}
	this->_data[this->geometry()->deviceIdForTransIdx(_xdcr_idx)][_xdcr_idx%NUM_TRANS_IN_UNIT] = (_amp << 8) + (_phase);
	this->_data[0][0] = 0xff00;
	this->_built = true;
}
