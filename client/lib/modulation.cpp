// File: modulation.cpp
// Project: lib
// Created Date: 11/06/2016
// Author: Seki Inoue
// -----
// Last Modified: 20/02/2020
// Modified By: Shun Suzuki (suzuki@hapis.k.u-tokyo.ac.jp)
// -----
// Copyright (c) 2016-2020 Hapis Lab. All rights reserved.
//

#include <stdio.h>
#define _USE_MATH_DEFINES
#include <math.h>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <limits>
#include <numeric>

#include "autd3.hpp"
#include "modulation.hpp"
#include "privdef.hpp"

#pragma region Util
static inline double sinc(double x) noexcept {
  if (fabs(x) < std::numeric_limits<double>::epsilon()) return 1;
  return sin(M_PI * x) / (M_PI * x);
}
#pragma endregion

#pragma region Modulation
autd::Modulation::Modulation() noexcept { this->sent = 0; }

constexpr int autd::Modulation::samplingFrequency() { return MOD_SAMPLING_FREQ; }

autd::ModulationPtr autd::Modulation::Create() { return CreateHelper<Modulation>(); }

autd::ModulationPtr autd::Modulation::Create(uint8_t amp) {
  auto mod = CreateHelper<Modulation>();
  mod->buffer.resize(MOD_FRAME_SIZE, amp);
  return mod;
}
#pragma endregion

#pragma region SineModulation
autd::ModulationPtr autd::SineModulation::Create(int freq, double amp, double offset) {
  assert(offset + 0.5 * amp <= 1.0 && offset - 0.5 * amp >= 0.0);

  auto mod = CreateHelper<SineModulation>();
  constexpr auto sf = autd::Modulation::samplingFrequency();
  freq = std::clamp(freq, 1, sf / 2);

  const auto d = std::gcd(sf, freq);

  const size_t N = MOD_BUF_SIZE / d;
  const size_t REP = freq / d;

  mod->buffer.resize(N, 0);

  for (size_t i = 0; i < N; i++) {
    auto tamp = fmod(static_cast<double>(2 * REP * i) / N, 2.0);
    tamp = tamp > 1.0 ? 2.0 - tamp : tamp;
    tamp = offset + (tamp - 0.5) * amp;
    mod->buffer.at(i) = static_cast<uint8_t>(tamp * 255.0);
  }

  return mod;
}
#pragma endregion

#pragma region SawModulation
autd::ModulationPtr autd::SawModulation::Create(int freq) {
  auto mod = CreateHelper<SawModulation>();
  constexpr auto sf = autd::Modulation::samplingFrequency();
  freq = std::clamp(freq, 1, sf / 2);

  const auto d = std::gcd(sf, freq);

  const size_t N = MOD_BUF_SIZE / d;
  const auto REP = freq / d;

  mod->buffer.resize(N, 0);

  for (size_t i = 0; i < N; i++) {
    auto tamp = fmod(static_cast<double>(REP * i) / N, 1.0);
    mod->buffer.at(i) = static_cast<uint8_t>(asin(tamp) / M_PI * 510.0);
  }

  return mod;
}
#pragma endregion

#pragma region RawPCMModulation
autd::ModulationPtr autd::RawPCMModulation::Create(std::string filename, double samplingFreq) {
  if (samplingFreq < std::numeric_limits<double>::epsilon()) samplingFreq = MOD_SAMPLING_FREQ;
  auto mod = CreateHelper<RawPCMModulation>();

  std::ifstream ifs;
  ifs.open(filename, std::ios::binary);

  if (ifs.fail()) throw new std::runtime_error("Error on opening file.");

  auto max_v = std::numeric_limits<double>::min();
  auto min_v = std::numeric_limits<double>::max();

  std::vector<int> tmp;
  char buf[sizeof(int)];
  while (ifs.read(buf, sizeof(int))) {
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
  std::vector<double> smpl_buf;
  const auto freqratio = autd::Modulation::samplingFrequency() / samplingFreq;
  smpl_buf.resize(tmp.size() * static_cast<size_t>(freqratio));
  for (size_t i = 0; i < smpl_buf.size(); i++) {
    smpl_buf.at(i) = (fmod(i / freqratio, 1.0) < 1 / freqratio) ? tmp.at(static_cast<int>(i / freqratio)) : 0.0;
  }

  // LPF
  const auto NTAP = 31;
  const auto cutoff = samplingFreq / 2 / autd::Modulation::samplingFrequency();
  std::vector<double> lpf(NTAP);
  for (int i = 0; i < NTAP; i++) {
    const auto t = i - NTAP / 2.0;
    lpf.at(i) = sinc(t * cutoff * 2.0);
  }

  std::vector<double> lpf_buf;
  lpf_buf.resize(smpl_buf.size(), 0);
  for (size_t i = 0; i < lpf_buf.size(); i++) {
    for (int j = 0; j < NTAP; j++) {
      lpf_buf.at(i) += smpl_buf.at((i - j + smpl_buf.size()) % smpl_buf.size()) * lpf.at(j);
    }
    max_v = std::max<double>(lpf_buf.at(i), max_v);
    min_v = std::min<double>(lpf_buf.at(i), min_v);
  }

  if (max_v == min_v) max_v = min_v + 1;
  mod->buffer.resize(lpf_buf.size(), 0);
  for (size_t i = 0; i < lpf_buf.size(); i++) {
    mod->buffer.at(i) = static_cast<uint8_t>(round(255.0 * (lpf_buf.at(i) - min_v) / (max_v - min_v)));
  }

  return mod;
}
#pragma endregion
