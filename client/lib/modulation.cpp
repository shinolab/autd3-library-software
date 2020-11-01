// File: modulation.cpp
// Project: lib
// Created Date: 11/06/2016
// Author: Seki Inoue
// -----
// Last Modified: 30/10/2020
// Modified By: Shun Suzuki (suzuki@hapis.k.u-tokyo.ac.jp)
// -----
// Copyright (c) 2016-2020 Hapis Lab. All rights reserved.
//

#include <stdio.h>
#define _USE_MATH_DEFINES
#include <math.h>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <numeric>

#include "configuration.hpp"
#include "consts.hpp"
#include "modulation.hpp"
#include "privdef.hpp"

using autd::MOD_BUF_SIZE;
using autd::MOD_SAMPLING_FREQ;

namespace autd::modulation {

#pragma region Util
static inline double sinc(double x) noexcept {
  if (fabs(x) < std::numeric_limits<double>::epsilon()) return 1;
  return sin(M_PI * x) / (M_PI * x);
}
#pragma endregion

#pragma region Modulation
Modulation::Modulation() noexcept { this->_sent = 0; }

ModulationPtr Modulation::Create(uint8_t amp) {
  auto mod = std::make_shared<Modulation>();
  mod->buffer.resize(1, amp);
  return mod;
}

void Modulation::Build(Configuration config) {}
#pragma endregion

#pragma region SineModulation
ModulationPtr SineModulation::Create(int freq, double amp, double offset) {
  auto mod = std::make_shared<SineModulation>();
  mod->_freq = freq;
  mod->_amp = amp;
  mod->_offset = offset;
  return mod;
}

void SineModulation::Build(Configuration config) {
  const auto sf = static_cast<int32_t>(config.mod_sampling_freq());
  const auto mod_buf_size = static_cast<int32_t>(config.mod_buf_size());

  const auto freq = std::clamp(this->_freq, 1, sf / 2);

  const auto d = std::gcd(sf, freq);

  const size_t N = (mod_buf_size / d) / (mod_buf_size / sf);
  const size_t REP = freq / d;

  this->buffer.resize(N, 0);

  for (size_t i = 0; i < N; i++) {
    auto tamp = fmod(static_cast<double>(2 * REP * i) / N, 2.0);
    tamp = tamp > 1.0 ? 2.0 - tamp : tamp;
    tamp = std::clamp(this->_offset + (tamp - 0.5) * this->_amp, 0.0, 1.0);
    this->buffer.at(i) = static_cast<uint8_t>(tamp * 255.0);
  }
}
#pragma endregion

#pragma region SquareModulation
ModulationPtr SquareModulation::Create(int freq, uint8_t low, uint8_t high) {
  auto mod = std::make_shared<SquareModulation>();
  mod->_freq = freq;
  mod->_low = low;
  mod->_high = high;
  return mod;
}

void SquareModulation::Build(Configuration config) {
  const auto sf = static_cast<int32_t>(config.mod_sampling_freq());
  const auto mod_buf_size = static_cast<int32_t>(config.mod_buf_size());

  const auto freq = std::clamp(this->_freq, 1, sf / 2);

  const auto d = std::gcd(sf, freq);

  const size_t N = (mod_buf_size / d) / (mod_buf_size / sf);

  this->buffer.resize(N, this->_high);
  std::memset(&this->buffer[0], this->_low, N / 2);
}
#pragma endregion

#pragma region SawModulation
ModulationPtr SawModulation::Create(int freq) {
  auto mod = std::make_shared<SawModulation>();
  mod->_freq = freq;
  return mod;
}

void SawModulation::Build(Configuration config) {
  const auto sf = static_cast<int32_t>(config.mod_sampling_freq());
  const auto mod_buf_size = static_cast<int32_t>(config.mod_buf_size());

  const auto freq = std::clamp(this->_freq, 1, sf / 2);

  const auto d = std::gcd(sf, freq);

  const size_t N = (mod_buf_size / d) / (mod_buf_size / sf);
  const auto REP = freq / d;

  this->buffer.resize(N, 0);

  for (size_t i = 0; i < N; i++) {
    auto tamp = fmod(static_cast<double>(REP * i) / N, 1.0);
    this->buffer.at(i) = static_cast<uint8_t>(asin(tamp) / M_PI * 510.0);
  }
}
#pragma endregion

#pragma region RawPCMModulation
ModulationPtr RawPCMModulation::Create(std::string filename, double sampling_freq) {
  auto mod = std::make_shared<RawPCMModulation>();
  mod->_sampling_freq = sampling_freq;

  std::ifstream ifs;
  ifs.open(filename, std::ios::binary);

  if (ifs.fail()) throw new std::runtime_error("Error on opening file.");

  std::vector<int32_t> tmp;
  char buf[sizeof(int32_t)];
  while (ifs.read(buf, sizeof(int32_t))) {
    int value;
    std::memcpy(&value, buf, sizeof(int32_t));
    tmp.push_back(value);
  }

  mod->_buf = tmp;

  return mod;
}

void RawPCMModulation::Build(Configuration config) {
  const auto mod_sf = static_cast<int32_t>(config.mod_sampling_freq());
  if (this->_sampling_freq < std::numeric_limits<double>::epsilon()) this->_sampling_freq = static_cast<double>(mod_sf);

  // up sampling
  std::vector<double> smpl_buf;
  const auto freqratio = mod_sf / _sampling_freq;
  smpl_buf.resize(this->_buf.size() * static_cast<size_t>(freqratio));
  for (size_t i = 0; i < smpl_buf.size(); i++) {
    smpl_buf.at(i) = (fmod(i / freqratio, 1.0) < 1 / freqratio) ? this->_buf.at(static_cast<int>(i / freqratio)) : 0.0;
  }

  // LPF
  const auto NTAP = 31;
  const auto cutoff = _sampling_freq / 2 / mod_sf;
  std::vector<double> lpf(NTAP);
  for (int i = 0; i < NTAP; i++) {
    const auto t = i - NTAP / 2.0;
    lpf.at(i) = sinc(t * cutoff * 2.0);
  }

  auto max_v = std::numeric_limits<double>::min();
  auto min_v = std::numeric_limits<double>::max();
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
  this->buffer.resize(lpf_buf.size(), 0);
  for (size_t i = 0; i < lpf_buf.size(); i++) {
    this->buffer.at(i) = static_cast<uint8_t>(round(255.0 * (lpf_buf.at(i) - min_v) / (max_v - min_v)));
  }
}
#pragma endregion

#pragma region WavModulation

namespace {
template <class T>
inline static T read_from_stream(std::ifstream &fsp) {
  char buf[sizeof(T)];
  if (!fsp.read(buf, sizeof(T))) throw new std::runtime_error("Invalid data length.");
  T v{};
  std::memcpy(&v, buf, sizeof(T));
  return v;
}
}  // namespace

ModulationPtr WavModulation::Create(std::string filename) {
  auto mod = std::make_shared<WavModulation>();

  std::ifstream fs;
  fs.open(filename, std::ios::binary);
  if (fs.fail()) throw new std::runtime_error("Error on opening file.");

  const uint32_t riff_tag = read_from_stream<uint32_t>(fs);
  if (riff_tag != 0x46464952u) throw new std::runtime_error("Invalid data format.");

  const uint32_t chunk_size = read_from_stream<uint32_t>(fs);

  const uint32_t wav_desc = read_from_stream<uint32_t>(fs);
  if (wav_desc != 0x45564157u) throw new std::runtime_error("Invalid data format.");

  const uint32_t fmt_desc = read_from_stream<uint32_t>(fs);
  if (fmt_desc != 0x20746d66u) throw new std::runtime_error("Invalid data format.");

  const uint32_t fmt_chunk_size = read_from_stream<uint32_t>(fs);
  if (fmt_chunk_size != 0x00000010u) throw new std::runtime_error("Invalid data format.");

  const uint16_t wave_fmt = read_from_stream<uint16_t>(fs);
  if (wave_fmt != 0x0001u) throw new std::runtime_error("Invalid data format. This supports only uncompressed linear PCM data.");

  const uint16_t channel = read_from_stream<uint16_t>(fs);
  if (channel != 0x0001u) throw new std::runtime_error("Invalid data format. This supports only monaural audio.");

  const uint32_t sampl_freq = read_from_stream<uint32_t>(fs);
  const uint32_t bytes_per_sec = read_from_stream<uint32_t>(fs);
  const uint16_t block_size = read_from_stream<uint16_t>(fs);
  const uint16_t bits_per_sampl = read_from_stream<uint16_t>(fs);

  const uint32_t data_desc = read_from_stream<uint32_t>(fs);
  if (data_desc != 0x61746164u) throw new std::runtime_error("Invalid data format.");

  const uint32_t data_chunk_size = read_from_stream<uint32_t>(fs);

  if ((bits_per_sampl != 8) && (bits_per_sampl != 16)) {
    throw new std::runtime_error("This only supports 8 or 16 bits per sampling data.");
  }

  std::vector<uint8_t> tmp;
  auto data_size = data_chunk_size / (bits_per_sampl / 8);
  for (size_t i = 0; i < data_size; i++) {
    if (bits_per_sampl == 8) {
      auto d = read_from_stream<uint8_t>(fs);
      tmp.push_back(d);
    } else if (bits_per_sampl == 16) {
      auto d32 = static_cast<int32_t>(read_from_stream<int16_t>(fs)) - std::numeric_limits<int16_t>::min();
      auto d8 = static_cast<uint8_t>(static_cast<float>(d32) / std::numeric_limits<uint16_t>::max() * std::numeric_limits<uint8_t>::max());
      tmp.push_back(d8);
    }
  }

  mod->_buf = tmp;
  mod->_sampl_freq = sampl_freq;

  return mod;
}

void WavModulation::Build(Configuration config) {
  const auto mod_sf = static_cast<int32_t>(config.mod_sampling_freq());
  const auto mod_buf_size = static_cast<int32_t>(config.mod_buf_size());

  // down sampling
  std::vector<uint8_t> smpl_buf;
  const double freq_ratio = static_cast<double>(mod_sf) / _sampl_freq;
  auto buffer_size = static_cast<size_t>(this->_buf.size() * freq_ratio);
  if (buffer_size > mod_buf_size) {
    const auto mod_play_length_max = mod_buf_size / mod_sf;
    std::cerr << "Wave data length is too long. The data is truncated to the first " << mod_play_length_max
              << ((mod_play_length_max == 1) ? " second." : " seconds.") << std::endl;
    buffer_size = mod_buf_size;
  }

  smpl_buf.resize(buffer_size);
  for (size_t i = 0; i < smpl_buf.size(); i++) {
    auto idx = static_cast<size_t>(i / freq_ratio);
    smpl_buf.at(i) = _buf.at(idx);
  }

  this->buffer = smpl_buf;
}
#pragma endregion
}  // namespace autd::modulation
