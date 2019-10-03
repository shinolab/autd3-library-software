/*
 * File: gain.cpp
 * Project: lib
 * Created Date: 01/06/2016
 * Author: Seki Inoue
 * -----
 * Last Modified: 04/09/2019
 * Modified By: Shun Suzuki (suzuki@hapis.k.u-tokyo.ac.jp)
 * -----
 * Copyright (c) 2016-2019 Hapis Lab. All rights reserved.
 *
 */

#include <iostream>
#include <string>
#include <map>
#include <vector>
#include <cassert> 

#include "core.hpp"
#include "autd3.hpp"
#include "controller.hpp"
#include "gain.hpp"
#include "privdef.hpp"

using namespace autd;
using namespace Eigen;

GainPtr Gain::Create()
{
	return CreateHelper<Gain>();
}

Gain::Gain() noexcept
{
	this->_built = false;
	this->_geometry = nullptr;
}

void Gain::build()
{
	if (this->built())
		return;
	auto geometry = this->geometry();
	assert(geometry != nullptr);

	for (int i = 0; i < geometry->numDevices(); i++)
	{
		this->_data[geometry->deviceIdForDeviceIdx(i)] = std::vector<uint16_t>(NUM_TRANS_IN_UNIT, 0x0000);
	}

	this->_built = true;
}

bool Gain::built() noexcept
{
	return this->_built;
}

GeometryPtr Gain::geometry() noexcept
{
	return this->_geometry;
}

void Gain::SetGeometry(const GeometryPtr& geometry) noexcept
{
	this->_geometry = geometry;
}

std::map<int, std::vector<uint16_t>> Gain::data()
{
	return this->_data;
}

GainPtr PlaneWaveGain::Create(Vector3f direction)
{
	auto ptr = CreateHelper<PlaneWaveGain>();
	ptr->_direction = direction;
	return ptr;
}

void PlaneWaveGain::build()
{
	if (this->built())
		return;
	auto geometry = this->geometry();
	assert(geometry != nullptr);

	this->_data.clear();
	const auto ndevice = geometry->numDevices();
	for (int i = 0; i < ndevice; i++)
	{
		this->_data[geometry->deviceIdForDeviceIdx(i)].resize(NUM_TRANS_IN_UNIT);
	}

	const auto ntrans = geometry->numTransducers();
	for (int i = 0; i < ntrans; i++)
	{
		this->_data[geometry->deviceIdForTransIdx(i)].at(i % NUM_TRANS_IN_UNIT) = 0xFF00;
	}

	this->_built = true;
}

GainPtr FocalPointGain::Create(Vector3f point)
{
	return FocalPointGain::Create(point, 255);
}

GainPtr FocalPointGain::Create(Vector3f point, uint8_t amp)
{
	auto gain = CreateHelper<FocalPointGain>();
	gain->_point = point;
	gain->_geometry = nullptr;
	gain->_amp = amp;
	return gain;
}

void FocalPointGain::build()
{
	if (this->built())
		return;

	auto geometry = this->geometry();
	assert(geometry != nullptr);

	this->_data.clear();

	const auto ndevice = geometry->numDevices();
	for (int i = 0; i < ndevice; i++)
	{
		this->_data[geometry->deviceIdForDeviceIdx(i)].resize(NUM_TRANS_IN_UNIT);
	}

	const auto ntrans = geometry->numTransducers();
	for (int i = 0; i < ntrans; i++)
	{
		const auto trp = geometry->position(i);
		const auto dist = (trp - this->_point).norm();
		const auto fphase = fmodf(dist, ULTRASOUND_WAVELENGTH) / ULTRASOUND_WAVELENGTH;
		const auto phase = static_cast<uint8_t>(round(255.0f * (1.0f - fphase)));
		uint8_t D, S;
		SignalDesign(this->_amp, phase, D, S);
		this->_data[geometry->deviceIdForTransIdx(i)].at(i % NUM_TRANS_IN_UNIT) = (static_cast<uint16_t>(D) << 8) + S;
	}

	this->_built = true;
}

GainPtr BesselBeamGain::Create(Vector3f point, Vector3f vec_n, float theta_z)
{
	return BesselBeamGain::Create(point, vec_n, theta_z, 255);
}

GainPtr BesselBeamGain::Create(Vector3f point, Vector3f vec_n, float theta_z, uint8_t amp)
{
	auto gain = CreateHelper<BesselBeamGain>();
	gain->_point = point;
	gain->_vec_n = vec_n;
	gain->_theta_z = theta_z;
	gain->_geometry = nullptr;
	gain->_amp = amp;
	return gain;
}

void BesselBeamGain::build()
{
	if (this->built())
		return;
	auto geometry = this->geometry();
	assert(geometry != nullptr);

	this->_data.clear();
	const auto ndevice = geometry->numDevices();
	for (int i = 0; i < ndevice; i++)
	{
		this->_data[geometry->deviceIdForDeviceIdx(i)].resize(NUM_TRANS_IN_UNIT);
	}
	const auto ntrans = geometry->numTransducers();

	const Vector3f _ez(0.f, 0.f, 1.0f);

	if (_vec_n.norm() > 0)
		_vec_n = _vec_n / _vec_n.norm();
	const Vector3f _v(_vec_n.y(), -_vec_n.x(), 0.f);

	auto _theta_w = asinf(_v.norm());

	for (int i = 0; i < ntrans; i++)
	{
		const auto trp = geometry->position(i);
		const auto _r = trp - this->_point;
		const Vector3f _v_x_r = _r.cross(_v);
		const Vector3f _R = cos(_theta_w) * _r + sin(_theta_w) * _v_x_r + _v.dot(_r) * (1.0f - cos(_theta_w)) * _v;
		const auto fphase = fmodf(sin(_theta_z) * sqrt(_R.x() * _R.x() + _R.y() * _R.y()) - cos(_theta_z) * _R.z(), ULTRASOUND_WAVELENGTH) / ULTRASOUND_WAVELENGTH;
		const auto phase = static_cast<uint8_t>(round(255.0f * (1.0f - fphase)));
		uint8_t D, S;
		SignalDesign(this->_amp, phase, D, S);
		this->_data[geometry->deviceIdForTransIdx(i)].at(i % NUM_TRANS_IN_UNIT) = (static_cast<uint16_t>(D) << 8) + S;
	}

	this->_built = true;
}

GainPtr CustomGain::Create(uint16_t* data, int dataLength)
{
	auto gain = CreateHelper<CustomGain>();
	gain->_rawdata.resize(dataLength);
	for (int i = 0; i < dataLength; i++)
		gain->_rawdata.at(i) = data[i];
	gain->_geometry = nullptr;
	return gain;
}

void CustomGain::build()
{
	if (this->built())
		return;
	auto geometry = this->geometry();
	assert(geometry != nullptr);

	this->_data.clear();
	const auto ndevice = geometry->numDevices();
	for (int i = 0; i < ndevice; i++)
	{
		this->_data[geometry->deviceIdForDeviceIdx(i)].resize(NUM_TRANS_IN_UNIT);
	}
	const auto ntrans = geometry->numTransducers();

	for (int i = 0; i < ntrans; i++)
	{
		const auto data = this->_rawdata[i];
		const auto amp = static_cast<uint8_t>(data >> 8);
		const auto phase = static_cast<uint8_t>(data);
		uint8_t D, S;
		SignalDesign(amp, phase, D, S);
		this->_data[geometry->deviceIdForTransIdx(i)].at(i % NUM_TRANS_IN_UNIT) = (static_cast<uint16_t>(D) << 8) + S;
	}

	this->_built = true;
}

GainPtr TransducerTestGain::Create(int idx, int amp, int phase)
{
	auto gain = CreateHelper<TransducerTestGain>();
	gain->_xdcr_idx = idx;
	gain->_amp = amp;
	gain->_phase = phase;
	return gain;
}

void TransducerTestGain::build()
{
	if (this->built())
		return;
	auto geometry = this->geometry();
	assert(geometry != nullptr);

	this->_data.clear();
	const auto ndevice = geometry->numDevices();
	for (int i = 0; i < ndevice; i++)
	{
		this->_data[geometry->deviceIdForDeviceIdx(i)].resize(NUM_TRANS_IN_UNIT);
	}

	uint8_t D, S;
	SignalDesign(this->_amp, this->_phase, D, S);
	this->_data[geometry->deviceIdForTransIdx(_xdcr_idx)].at(_xdcr_idx % NUM_TRANS_IN_UNIT) = (static_cast<uint16_t>(D) << 8) + S;

	this->_built = true;
}
