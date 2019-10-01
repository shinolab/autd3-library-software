﻿/*
 * File: modulation.hpp
 * Project: include
 * Created Date: 04/11/2018
 * Author: Shun Suzuki
 * -----
 * Last Modified: 04/09/2019
 * Modified By: Shun Suzuki (suzuki@hapis.k.u-tokyo.ac.jp)
 * -----
 * Copyright (c) 2018-2019 Hapis Lab. All rights reserved.
 * 
 */

#pragma once

#include <memory>
#include <vector>
#include "core.hpp"

namespace autd
{
class Modulation;

#if DLL_FOR_CAPI
using ModulationPtr = Modulation *;
#else
using ModulationPtr = std::shared_ptr<Modulation>;
#endif

class Modulation
{
	friend class Controller;
	//friend class internal::Link;
public:
	Modulation() noexcept;
	static ModulationPtr Create();
	static ModulationPtr Create(uint8_t amp);
	constexpr static float samplingFrequency();
	bool loop;
	std::vector<uint8_t> buffer;

private:
	size_t sent;
};

class SineModulation : public Modulation
{
public:
	static ModulationPtr Create(float freq, float amp = 1.0f, float offset = 0.5f);
};

class SawModulation : public Modulation
{
public:
	static ModulationPtr Create(float freq);
};

class RawPCMModulation : public Modulation
{
public:
	static ModulationPtr Create(std::string filename, float samplingFreq = 0.0f);
};
} // namespace autd