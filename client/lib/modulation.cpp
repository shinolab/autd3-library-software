// File: modulation.cpp
// Project: lib
// Created Date: 11/06/2016
// Author: Seki Inoue
// -----
// Last Modified: 25/12/2020
// Modified By: Shun Suzuki (suzuki@hapis.k.u-tokyo.ac.jp)
// -----
// Copyright (c) 2016-2020 Hapis Lab. All rights reserved.
//

#define _USE_MATH_DEFINES  // NOLINT

#include "modulation.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <numeric>

#include "configuration.hpp"

using autd::MOD_BUF_SIZE;
using autd::MOD_SAMPLING_FREQ;

namespace autd::modulation {

#pragma region Util
inline double Sinc(const double x) noexcept {
  if (fabs(x) < std::numeric_limits<double>::epsilon()) return 1;
  return sin(M_PI * x) / (M_PI * x);
}
#pragma endregion

#pragma region Modulation
Modulation::Modulation() noexcept { this->_sent = 0; }

ModulationPtr Modulation::Create(const uint8_t amp) {
  auto mod = std::make_shared<Modulation>();
  mod->buffer.resize(1, amp);
  return mod;
}

void Modulation::Build(Configuration config) {}

size_t& Modulation::sent() { return _sent; }
#pragma endregion

#pragma region SineModulation
ModulationPtr SineModulation::Create(const int freq, const double amp, const double offset) {
  ModulationPtr mod = std::make_shared<SineModulation>(freq, amp, offset);
  return mod;
}

void SineModulation::Build(const Configuration config) {
  const auto sf = static_cast<int32_t>(config.mod_sampling_freq());
  const auto mod_buf_size = static_cast<int32_t>(config.mod_buf_size());

  const auto freq = std::clamp(this->_freq, 1, sf / 2);

  const auto d = std::gcd(sf, freq);

  const size_t n = mod_buf_size / d / (mod_buf_size / sf);
  const size_t rep = freq / d;

  this->buffer.resize(n, 0);

  for (size_t i = 0; i < n; i++) {
    auto tamp = fmod(static_cast<double>(2 * rep * i) / static_cast<double>(n), 2.0);
    tamp = tamp > 1.0 ? 2.0 - tamp : tamp;
    tamp = std::clamp(this->_offset + (tamp - 0.5) * this->_amp, 0.0, 1.0);
    this->buffer.at(i) = static_cast<uint8_t>(tamp * 255.0);
  }
}
#pragma endregion

#pragma region SquareModulation
ModulationPtr SquareModulation::Create(int freq, uint8_t low, uint8_t high) {
  ModulationPtr mod = std::make_shared<SquareModulation>(freq, low, high);
  return mod;
}

void SquareModulation::Build(const Configuration config) {
  const auto sf = static_cast<int32_t>(config.mod_sampling_freq());
  const auto mod_buf_size = static_cast<int32_t>(config.mod_buf_size());

  const auto freq = std::clamp(this->_freq, 1, sf / 2);

  const auto d = std::gcd(sf, freq);

  const size_t n = mod_buf_size / d / (mod_buf_size / sf);

  this->buffer.resize(n, this->_high);
  std::memset(&this->buffer[0], this->_low, n / 2);
}
#pragma endregion

#pragma region SawModulation
ModulationPtr SawModulation::Create(const int freq) {
  ModulationPtr mod = std::make_shared<SawModulation>(freq);
  return mod;
}

void SawModulation::Build(const Configuration config) {
  const auto sf = static_cast<int32_t>(config.mod_sampling_freq());
  const auto mod_buf_size = static_cast<int32_t>(config.mod_buf_size());

  const auto freq = std::clamp(this->_freq, 1, sf / 2);

  const auto d = std::gcd(sf, freq);

  const size_t n = mod_buf_size / d / (mod_buf_size / sf);
  const auto rep = freq / d;

  this->buffer.resize(n, 0);

  for (size_t i = 0; i < n; i++) {
    const auto tamp = fmod(static_cast<double>(rep * i) / static_cast<double>(n), 1.0);
    this->buffer.at(i) = static_cast<uint8_t>(asin(tamp) / M_PI * 510.0);
  }
}
#pragma endregion

#pragma region RawPCMModulation
ModulationPtr RawPCMModulation::Create(const std::string& filename, const double sampling_freq) {
  std::ifstream ifs;
  ifs.open(filename, std::ios::binary);

  if (ifs.fail()) throw std::runtime_error("Error on opening file.");

  std::vector<int32_t> tmp;
  char buf[sizeof(int32_t)];
  while (ifs.read(buf, sizeof(int32_t))) {
    int value;
    std::memcpy(&value, buf, sizeof(int32_t));
    tmp.push_back(value);
  }

  ModulationPtr mod = std::make_shared<RawPCMModulation>(sampling_freq, tmp);
  return mod;
}

auto RawPCMModulation::Build(const Configuration config) -> void {
  const auto mod_sf = static_cast<int32_t>(config.mod_sampling_freq());
  if (this->_sampling_freq < std::numeric_limits<double>::epsilon()) this->_sampling_freq = static_cast<double>(mod_sf);

  // up sampling
  std::vector<double> sample_buf;
  const auto freq_ratio = mod_sf / _sampling_freq;
  sample_buf.resize(this->_buf.size() * static_cast<size_t>(freq_ratio));
  for (size_t i = 0; i < sample_buf.size(); i++) {
    const auto v = static_cast<double>(i) / freq_ratio;
    sample_buf.at(i) = fmod(v, 1.0) < 1 / freq_ratio ? this->_buf.at(static_cast<int>(v)) : 0.0;
  }

  // LPF
  const auto num_tap = 31;
  const auto cutoff = _sampling_freq / 2 / mod_sf;
  std::vector<double> lpf(num_tap);
  for (auto i = 0; i < num_tap; i++) {
    const auto t = i - num_tap / 2.0;
    lpf.at(i) = Sinc(t * cutoff * 2.0);
  }

  auto max_v = std::numeric_limits<double>::min();
  auto min_v = std::numeric_limits<double>::max();
  std::vector<double> lpf_buf;
  lpf_buf.resize(sample_buf.size(), 0);
  for (size_t i = 0; i < lpf_buf.size(); i++) {
    for (auto j = 0; j < num_tap; j++) {
      lpf_buf.at(i) += sample_buf.at((i - j + sample_buf.size()) % sample_buf.size()) * lpf.at(j);
    }
    max_v = std::max<double>(lpf_buf.at(i), max_v);
    min_v = std::min<double>(lpf_buf.at(i), min_v);
  }

  if (max_v - min_v < std::numeric_limits<double>::epsilon()) max_v = min_v + 1;
  this->buffer.resize(lpf_buf.size(), 0);
  for (size_t i = 0; i < lpf_buf.size(); i++) {
    this->buffer.at(i) = static_cast<uint8_t>(round(255.0 * (lpf_buf.at(i) - min_v) / (max_v - min_v)));
  }
}
#pragma endregion

#pragma region WavModulation

namespace {
template <class T>
T ReadFromStream(std::ifstream& fsp) {
  char buf[sizeof(T)];
  if (!fsp.read(buf, sizeof(T))) throw std::runtime_error("Invalid data length.");
  T v{};
  std::memcpy(&v, buf, sizeof(T));
  return v;
}
}  // namespace

ModulationPtr WavModulation::Create(const std::string& filename) {
  std::ifstream fs;
  fs.open(filename, std::ios::binary);
  if (fs.fail()) throw std::runtime_error("Error on opening file.");

  const auto riff_tag = ReadFromStream<uint32_t>(fs);
  if (riff_tag != 0x46464952u) throw std::runtime_error("Invalid data format.");

  [[maybe_unused]] const auto chunk_size = ReadFromStream<uint32_t>(fs);

  const auto wav_desc = ReadFromStream<uint32_t>(fs);
  if (wav_desc != 0x45564157u) throw std::runtime_error("Invalid data format.");

  const auto fmt_desc = ReadFromStream<uint32_t>(fs);
  if (fmt_desc != 0x20746d66u) throw std::runtime_error("Invalid data format.");

  const auto fmt_chunk_size = ReadFromStream<uint32_t>(fs);
  if (fmt_chunk_size != 0x00000010u) throw std::runtime_error("Invalid data format.");

  const auto wave_fmt = ReadFromStream<uint16_t>(fs);
  if (wave_fmt != 0x0001u) throw std::runtime_error("Invalid data format. This supports only uncompressed linear PCM data.");

  const auto channel = ReadFromStream<uint16_t>(fs);
  if (channel != 0x0001u) throw std::runtime_error("Invalid data format. This supports only monaural audio.");

  const auto sample_freq = ReadFromStream<uint32_t>(fs);
  [[maybe_unused]] const auto bytes_per_sec = ReadFromStream<uint32_t>(fs);
  [[maybe_unused]] const auto block_size = ReadFromStream<uint16_t>(fs);
  const auto bits_per_sample = ReadFromStream<uint16_t>(fs);

  const auto data_desc = ReadFromStream<uint32_t>(fs);
  if (data_desc != 0x61746164u) throw std::runtime_error("Invalid data format.");

  const auto data_chunk_size = ReadFromStream<uint32_t>(fs);

  if (bits_per_sample != 8 && bits_per_sample != 16) {
    throw std::runtime_error("This only supports 8 or 16 bits per sampling data.");
  }

  std::vector<uint8_t> tmp;
  const auto data_size = data_chunk_size / (bits_per_sample / 8);
  for (size_t i = 0; i < data_size; i++) {
    if (bits_per_sample == 8) {
      auto d = ReadFromStream<uint8_t>(fs);
      tmp.push_back(d);
    } else if (bits_per_sample == 16) {
      const auto d32 = static_cast<int32_t>(ReadFromStream<int16_t>(fs)) - std::numeric_limits<int16_t>::min();
      auto d8 = static_cast<uint8_t>(static_cast<float>(d32) / std::numeric_limits<uint16_t>::max() * std::numeric_limits<uint8_t>::max());
      tmp.push_back(d8);
    }
  }

  ModulationPtr mod = std::make_shared<WavModulation>(sample_freq, tmp);
  return mod;
}

void WavModulation::Build(const Configuration config) {
  const auto mod_sf = static_cast<int32_t>(config.mod_sampling_freq());
  const auto mod_buf_size = static_cast<size_t>(config.mod_buf_size());

  // down sampling
  std::vector<uint8_t> sample_buf;
  const auto freq_ratio = static_cast<double>(mod_sf) / _sampling_freq;
  auto buffer_size = static_cast<size_t>(static_cast<double>(this->_buf.size()) * freq_ratio);
  if (buffer_size > mod_buf_size) {
    const auto mod_play_length_max = mod_buf_size / mod_sf;
    std::cerr << "Wave data length is too long. The data is truncated to the first " << mod_play_length_max
              << (mod_play_length_max == 1 ? " second." : " seconds.") << std::endl;
    buffer_size = mod_buf_size;
  }

  sample_buf.resize(buffer_size);
  for (size_t i = 0; i < sample_buf.size(); i++) {
    const auto idx = static_cast<size_t>(static_cast<double>(i) / freq_ratio);
    sample_buf.at(i) = _buf.at(idx);
  }

  this->buffer = sample_buf;
}
#pragma endregion
}  // namespace autd::modulation
