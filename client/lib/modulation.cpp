//
//  modulation.cpp
//  autd3
//
//  Created by Seki Inoue on 6/11/16.
//  Changed by Shun Suzuki on 02/07/2018.
//
//

#include <stdio.h>
#define _USE_MATH_DEFINES
#include <math.h>
#include <cmath>
#include <limits>
#include <fstream>
#include "autd3.hpp"
#include "privdef.hpp"

autd::Modulation::Modulation() {
	this->sent = 0;
	this->loop = true;
}


const float autd::Modulation::samplingFrequency() {
	return MOD_SAMPLING_FREQ;
}

autd::ModulationPtr autd::Modulation::Create() {
	return  CreateHelper<Modulation>();
}

autd::ModulationPtr autd::Modulation::Create(uint8_t amp) {
	auto mod = CreateHelper<Modulation>();
	mod->buffer.resize(1, amp);
	return mod;
}


autd::ModulationPtr autd::SineModulation::Create(float freq, float amp, float offset) {
	auto mod = CreateHelper<SineModulation>();

	freq = std::min<float>(mod->samplingFrequency() / 2, std::max<float>(1.0, freq));

	int T = (int)round(1.0 / freq * mod->samplingFrequency());
	mod->buffer.resize(T, 0);
	for (int i = 0; i < T; i++) {
		float tamp = 255.0f*offset + 127.5f*amp*cosf(2.0f * M_PIf * i / T);
		mod->buffer[i] = (uint8_t)floor(fmin(fmaxf(tamp, 0.0f), 255.0f));
		if (mod->buffer[i] == 0) mod->buffer[i] = 1;
	}
	mod->loop = true;
	return mod;
}

autd::ModulationPtr autd::SawModulation::Create(float freq) {
	auto mod = CreateHelper<SawModulation>();

	freq = std::min<float>(mod->samplingFrequency() / 2, std::max<float>(1.0, freq));

	int T = (int)round(1.0 / freq * mod->samplingFrequency());
	mod->buffer.resize(T, 0);
	for (int i = 0; i < T; i++) {
		float amp = 255.0f * i / T;
		mod->buffer[i] = (uint8_t)floor(fmaxf(amp, 0.0f));
	}
	mod->loop = true;
	return mod;
}

inline float sinc(float x) {
	if (fabs(x) < std::numeric_limits<float>::epsilon()) return 1;
	return sinf(M_PIf*x) / (M_PIf*x);
}

autd::ModulationPtr autd::RawPCMModulation::Create(std::string filename, float samplingFreq) {
	if (samplingFreq < std::numeric_limits<float>::epsilon()) samplingFreq = MOD_SAMPLING_FREQ;
	auto mod = CreateHelper<RawPCMModulation>();

	std::ifstream ifs;
	ifs.open(filename, std::ios::binary);

	if (ifs.fail()) throw new std::runtime_error("Error on opening file.");

	float max_v = std::numeric_limits<float>::min();
	float min_v = std::numeric_limits<float>::max();

	std::vector<int> tmp;
	int value;
	char buf[sizeof(int)];
	while (ifs.read(buf, sizeof(buf)))
	{
		memcpy(&value, buf, sizeof(value));
		tmp.push_back(value);
	}

	// 少なくともVS2017ではこのコードが動かない
	// 具体的には永遠にvに0が入る
/*
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
	const float freqratio = mod->samplingFrequency() / samplingFreq;
	smpl_buf.resize(tmp.size() * (size_t)freqratio);
	for (size_t i = 0; i < smpl_buf.size(); i++) {
		smpl_buf[i] = (fmod(i / freqratio, 1.0) < 1 / freqratio) ? tmp[(int)(i / freqratio)] : 0.0f;
	}

	// LPF
	// TODO: window function
	const int NTAP = 31;
	float cutoff = samplingFreq / 2 / mod->samplingFrequency();
	float lpf[NTAP];
	for (int i = 0; i < NTAP; i++) {
		float t = i - NTAP / 2.0f;
		lpf[i] = sinc(t*cutoff*2.0f);
	}

	std::vector<float> lpf_buf;
	lpf_buf.resize(smpl_buf.size(), 0);
	for (size_t i = 0; i < lpf_buf.size(); i++) {
		for (int j = 0; j < NTAP; j++) {
			lpf_buf[i] += smpl_buf[(i - j + smpl_buf.size()) % smpl_buf.size()] * lpf[j];
		}
		max_v = std::max<float>(lpf_buf[i], max_v);
		min_v = std::min<float>(lpf_buf[i], min_v);
	}

	if (max_v == min_v) max_v = min_v + 1;
	mod->buffer.resize(lpf_buf.size(), 0);
	for (size_t i = 0; i < lpf_buf.size(); i++) {
		mod->buffer[i] = (uint8_t)round(255.0 * (lpf_buf[i] - min_v) / (max_v - min_v));
	}
	mod->loop = true;

	return mod;
}
