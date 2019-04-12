//
//  gain.cpp
//  autd3
//
//  Created by Seki Inoue on 6/1/16.
//  Modified by Shun Suzuki on 02/07/2018.
//  Modified by Shun Suzuki on 04/11/2018.
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

#include "core.hpp"
#include "autd3.hpp"
#include "controller.hpp"
#include "gain.hpp"
#include "privdef.hpp"

using namespace autd;
using namespace Eigen;

GainPtr Gain::Create() {
	return CreateHelper<Gain>();
}

Gain::Gain() noexcept {
	this->_built = false;
	this->_fix = false;
	this->_geometry = nullptr;
}

void Gain::build() {
	if (this->built()) return;

	auto geo = this->geometry();
	if (geo == nullptr) BOOST_ASSERT_MSG(false, "Geometry is required to build Gain");

	for (int i = 0; i < geo->numDevices(); i++) {
		this->_data[geo->deviceIdForDeviceIdx(i)] = std::vector<uint16_t>(NUM_TRANS_IN_UNIT, 0x0000);
	}
}

bool Gain::built() {
	return this->_built;
}

void Gain::SetGeometry(const GeometryPtr& geometry) noexcept {
	this->_geometry = geometry;
}

inline void ConvertAmpPhase(uint8_t amp_i, uint8_t phase_i, uint8_t& amp_o, uint8_t& phase_o) noexcept {
	auto d = asin(amp_i / 255.0) / M_PIf;  // duty (0 ~ 0.5)
	amp_o = static_cast<uint8_t>(511 * d);
	phase_o = static_cast<uint8_t>(static_cast<int>(phase_i + 256 - 128 * d) % 256);
}

void Gain::Fix() noexcept {
	this->_fix = true;
}

void Gain::FixImpl() {
	auto geo = this->geometry();
	for (int i = 0; i < geo->numDevices(); i++) {
		std::vector<uint16_t>* vec = &this->_data.at(geo->deviceIdForDeviceIdx(i));
		for (size_t j = 0; j < vec->size(); j++)
		{
			auto amp = static_cast<uint8_t>((*vec).at(j) >> 8);
			auto phase = static_cast<uint8_t>((*vec).at(j));
			uint8_t amp_fix, phase_fix;
			ConvertAmpPhase(amp, phase, amp_fix, phase_fix);
			(*vec).at(j) = (static_cast<uint16_t>(amp_fix) << 8) + phase_fix;
		}
	}
}

GeometryPtr Gain::geometry() noexcept {
	return this->_geometry;
}

std::map<int, std::vector<uint16_t> > Gain::data() {
	return this->_data;
}

GainPtr PlaneWaveGain::Create(Vector3f direction) {
	auto ptr = CreateHelper<PlaneWaveGain>();
	ptr->_direction = direction;
	return ptr;
}

void PlaneWaveGain::build() {
	if (this->built()) return;

	auto geo = this->geometry();
	if (geo == nullptr) BOOST_ASSERT_MSG(false, "Geometry is required to build Gain");

	this->_data.clear();
	const auto ndevice = geo->numDevices();
	for (int i = 0; i < ndevice; i++) {
		this->_data[geo->deviceIdForDeviceIdx(i)].resize(NUM_TRANS_IN_UNIT);
	}

	const auto ntrans = geo->numTransducers();
	for (int i = 0; i < ntrans; i++) {
		this->_data[geo->deviceIdForTransIdx(i)].at(i % NUM_TRANS_IN_UNIT) = 0xFF00;
	}

	this->_built = true;
}

GainPtr FocalPointGain::Create(Vector3f point) {
	return FocalPointGain::Create(point, 255);
}

GainPtr FocalPointGain::Create(Vector3f point, uint8_t amp) {
	auto gain = CreateHelper<FocalPointGain>();
	gain->_point = point;
	gain->_geometry = nullptr;
	gain->_amp = amp;
	return gain;
}

void FocalPointGain::build() {
	if (this->built()) return;
	auto geo = this->geometry();
	if (geo == nullptr) BOOST_ASSERT_MSG(false, "Geometry is required to build Gain");

	this->_data.clear();

	const auto ndevice = geo->numDevices();
	for (int i = 0; i < ndevice; i++) {
		this->_data[geo->deviceIdForDeviceIdx(i)].resize(NUM_TRANS_IN_UNIT);
	}

	const auto ntrans = geo->numTransducers();
	for (int i = 0; i < ntrans; i++) {
		const auto trp = geo->position(i);
		const auto dist = (trp - this->_point).norm();
		const auto fphase = fmodf(dist, ULTRASOUND_WAVELENGTH) / ULTRASOUND_WAVELENGTH;
		const auto amp = this->_amp;
		const auto phase = static_cast<uint8_t>(round(255.0f * (1.0f - fphase)));
		this->_data[geo->deviceIdForTransIdx(i)].at(i % NUM_TRANS_IN_UNIT) = (static_cast<uint16_t>(amp) << 8) + phase;
	}

	this->_built = true;
}

GainPtr BesselBeamGain::Create(Vector3f point, Vector3f vec_n, float theta_z) {
	auto gain = CreateHelper<BesselBeamGain>();
	gain->_point = point;
	gain->_vec_n = vec_n;
	gain->_theta_z = theta_z;
	gain->_geometry = nullptr;
	return gain;
}

void BesselBeamGain::build() {
	if (this->built()) return;
	auto geo = this->geometry();
	if (geo == nullptr) BOOST_ASSERT_MSG(false, "Geometry is required to build Gain");

	this->_data.clear();
	const auto ndevice = geo->numDevices();
	for (int i = 0; i < ndevice; i++) {
		this->_data[geo->deviceIdForDeviceIdx(i)].resize(NUM_TRANS_IN_UNIT);
	}
	const auto ntrans = geo->numTransducers();

	const Vector3f _ez(0.f, 0.f, 1.0f);

	if (_vec_n.norm() > 0) _vec_n = _vec_n / _vec_n.norm();
	const Vector3f _v(_vec_n.y(), -_vec_n.x(), 0.f);

	auto _theta_w = asinf(_v.norm());

	for (int i = 0; i < ntrans; i++) {
		const auto trp = geo->position(i);
		const auto _r = trp - this->_point;
		const Vector3f _v_x_r = _r.cross(_v);
		const Vector3f _R = cos(_theta_w) * _r + sin(_theta_w) * _v_x_r + _v.dot(_r) * (1.0f - cos(_theta_w)) * _v;
		const auto fphase = fmodf(sin(_theta_z) * sqrt(_R.x() * _R.x() + _R.y() * _R.y()) - cos(_theta_z) * _R.z(), ULTRASOUND_WAVELENGTH) / ULTRASOUND_WAVELENGTH;
		const auto amp = 0xff;
		const auto phase = static_cast<uint8_t>(round(255.0f * (1.0f - fphase)));
		this->_data[geo->deviceIdForTransIdx(i)].at(i % NUM_TRANS_IN_UNIT) = (static_cast<uint16_t>(amp) << 8) + phase;
	}

	this->_built = true;
}

GainPtr CustomGain::Create(uint16_t * data, int dataLength) {
	auto gain = CreateHelper<CustomGain>();;
	gain->_rawdata.resize(dataLength);
	for (int i = 0; i < dataLength; i++) gain->_rawdata.at(i) = data[i];
	gain->_geometry = nullptr;
	return gain;
}

void CustomGain::build() {
	if (this->built()) return;
	auto geo = this->geometry();
	if (geo == nullptr) BOOST_ASSERT_MSG(false, "Geometry is required to build Gain");

	this->_data.clear();
	const auto ndevice = geo->numDevices();
	for (int i = 0; i < ndevice; i++) {
		this->_data[geo->deviceIdForDeviceIdx(i)].resize(NUM_TRANS_IN_UNIT);
	}
	const auto ntrans = geo->numTransducers();

	for (int i = 0; i < ntrans; i++) {
		this->_data[geo->deviceIdForTransIdx(i)].at(i % NUM_TRANS_IN_UNIT) = this->_rawdata[i];
	}

	this->_built = true;
}

GainPtr TransducerTestGain::Create(int idx, int amp, int phase) {
	auto gain = CreateHelper<TransducerTestGain>();
	gain->_xdcr_idx = idx;
	gain->_amp = amp;
	gain->_phase = phase;
	return gain;
}

void TransducerTestGain::build() {
	if (this->built()) return;
	auto geo = this->geometry();
	if (geo == nullptr) BOOST_ASSERT_MSG(false, "Geometry is required to build Gain");

	this->_data.clear();
	const auto ndevice = geo->numDevices();
	for (int i = 0; i < ndevice; i++) {
		this->_data[geo->deviceIdForDeviceIdx(i)].resize(NUM_TRANS_IN_UNIT);
	}
	this->_data[geo->deviceIdForTransIdx(_xdcr_idx)].at(_xdcr_idx % NUM_TRANS_IN_UNIT) = (_amp << 8) + (_phase);
	this->_data[0].at(0) = 0xff00;
	this->_built = true;
}
