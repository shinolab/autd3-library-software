/*
 * File: modulation.cpp
 * Project: lib
 * Created Date: 11/06/2016
 * Author: Seki Inoue
 * -----
 * Last Modified: 19/10/2019
 * Modified By: Shun Suzuki (suzuki@hapis.k.u-tokyo.ac.jp)
 * -----
 * Copyright (c) 2016-2019 Hapis Lab. All rights reserved.
 *
 */

#include <stdio.h>
#define _USE_MATH_DEFINES
#include <math.h>
#include <cmath>
#include <limits>
#include <fstream>

#include "autd3.hpp"
#include "modulation.hpp"
#include "privdef.hpp"

#pragma region Util
inline float sinc(float x) noexcept
{
	if (fabs(x) < std::numeric_limits<float>::epsilon())
		return 1;
	return sinf(M_PIf * x) / (M_PIf * x);
}
#if _WINDOWS
template <typename T>
T clamp(T v, T min, T max)
{
	return (v < min) ? min : (v > max) ? max : v;
}
#endif
int gcd(int u, int V)
{
	const auto t = u % V;
	return (t == 0) ? V : gcd(V, t);
}
#pragma endregion

#pragma region Modulation
autd::Modulation::Modulation() noexcept
{
	this->sent = 0;
}

constexpr int autd::Modulation::samplingFrequency()
{
	return MOD_SAMPLING_FREQ;
}

autd::ModulationPtr autd::Modulation::Create()
{
	return CreateHelper<Modulation>();
}

autd::ModulationPtr autd::Modulation::Create(uint8_t amp)
{
	auto mod = CreateHelper<Modulation>();
	mod->buffer.resize(MOD_FRAME_SIZE, amp);
	return mod;
}
#pragma endregion

#pragma region SineModulation
autd::ModulationPtr autd::SineModulation::Create(int freq, float amp, float offset)
{
	assert(offset + 0.5f * amp <= 1.0f && offset - 0.5f * amp >= 0.0f);

	auto mod = CreateHelper<SineModulation>();
	constexpr auto sf = autd::Modulation::samplingFrequency();
	freq = clamp(freq, 1, sf / 2);

	const auto d = gcd(sf, freq);

	const auto N = MOD_BUF_SIZE / d;
	const auto REP = freq / d;

	mod->buffer.resize(N, 0);

	for (size_t i = 0; i < N; i++)
	{
		auto tamp = fmodf(static_cast<float>(2 * REP * i) / N, 2.0f);
		tamp = tamp > 1.0f ? 2.0f - tamp : tamp;
		tamp = offset + (tamp - 0.5f) * amp;
		mod->buffer.at(i) = static_cast<uint8_t>(tamp * 255.0f);
	}

	return mod;
}
#pragma endregion

#pragma region SawModulation
autd::ModulationPtr autd::SawModulation::Create(int freq)
{
	auto mod = CreateHelper<SawModulation>();
	constexpr auto sf = autd::Modulation::samplingFrequency();
	freq = clamp(freq, 1, sf / 2);

	const auto d = gcd(sf, freq);

	const auto N = MOD_BUF_SIZE / d;
	const auto REP = freq / d;

	mod->buffer.resize(N, 0);

	for (size_t i = 0; i < N; i++)
	{
		auto tamp = fmodf(static_cast<float>(REP * i) / N, 1.0f);
		mod->buffer.at(i) = static_cast<uint8_t>(asin(tamp) / M_PIf * 510.0f);
	}

	return mod;
}
#pragma endregion

#pragma region RawPCMModulation
autd::ModulationPtr autd::RawPCMModulation::Create(std::string filename, float samplingFreq)
{
	if (samplingFreq < std::numeric_limits<float>::epsilon())
		samplingFreq = MOD_SAMPLING_FREQ;
	auto mod = CreateHelper<RawPCMModulation>();

	std::ifstream ifs;
	ifs.open(filename, std::ios::binary);

	if (ifs.fail())
		throw new std::runtime_error("Error on opening file.");

	auto max_v = std::numeric_limits<float>::min();
	auto min_v = std::numeric_limits<float>::max();

	std::vector<int> tmp;
	char buf[sizeof(int)];
	while (ifs.read(buf, sizeof(int)))
	{
		int value;
		memcpy(&value, buf, sizeof(int));
		tmp.push_back(value);
	}
	/*
		以下が元の実装
		少なくともVS2017ではこのコードが動かない
		具体的には永遠にvに0が入る
		do {
			short v = 0;
			ifs >> v;
			tmp.push_back(v);
		} while (!ifs.eof());
	*/

	// up sampling
	// TODO: impl. down sampling
	// TODO: efficient memory management

	std::vector<float> smpl_buf;
	const auto freqratio = autd::Modulation::samplingFrequency() / samplingFreq;
	smpl_buf.resize(tmp.size() * static_cast<size_t>(freqratio));
	for (size_t i = 0; i < smpl_buf.size(); i++)
	{
		smpl_buf.at(i) = (fmod(i / freqratio, 1.0) < 1 / freqratio) ? tmp.at(static_cast<int>(i / freqratio)) : 0.0f;
	}

	// LPF
	// TODO: window function
	const auto NTAP = 31;
	const auto cutoff = samplingFreq / 2 / autd::Modulation::samplingFrequency();
	std::vector<float> lpf(NTAP);
	for (int i = 0; i < NTAP; i++)
	{
		const auto t = i - NTAP / 2.0f;
		lpf.at(i) = sinc(t * cutoff * 2.0f);
	}

	std::vector<float> lpf_buf;
	lpf_buf.resize(smpl_buf.size(), 0);
	for (size_t i = 0; i < lpf_buf.size(); i++)
	{
		for (int j = 0; j < NTAP; j++)
		{
			lpf_buf.at(i) += smpl_buf.at((i - j + smpl_buf.size()) % smpl_buf.size()) * lpf.at(j);
		}
		max_v = std::max<float>(lpf_buf.at(i), max_v);
		min_v = std::min<float>(lpf_buf.at(i), min_v);
	}

	if (max_v == min_v)
		max_v = min_v + 1;
	mod->buffer.resize(lpf_buf.size(), 0);
	for (size_t i = 0; i < lpf_buf.size(); i++)
	{
		mod->buffer.at(i) = static_cast<uint8_t>(round(255.0f * (lpf_buf.at(i) - min_v) / (max_v - min_v)));
	}

	return mod;
}
#pragma endregion
