// File: from_file.cpp
// Project: from_file
// Created Date: 20/02/2021
// Author: Shun Suzuki
// -----
// Last Modified: 06/03/2021
// Modified By: Shun Suzuki (suzuki@hapis.k.u-tokyo.ac.jp)
// -----
// Copyright (c) 2021 Hapis Lab. All rights reserved.
//

#include "modulation/from_file.hpp"

#include <cstring>
#include <fstream>
#include <iostream>

namespace autd::modulation {

#pragma region RawPCMModulation
ModulationPtr RawPCMModulation::Create(const std::string& filename, const Float sampling_freq) {
  std::ifstream ifs;
  ifs.open(filename, std::ios::binary);

  if (ifs.fail()) throw std::runtime_error("Error on opening file.");

  std::vector<int32_t> tmp;
  char buf[sizeof(int32_t)];
  while (ifs.read(buf, sizeof(int32_t))) {
    int value;
    std::memcpy(&value, buf, sizeof(int32_t));
    tmp.emplace_back(value);
  }

  ModulationPtr mod = std::make_shared<RawPCMModulation>(sampling_freq, tmp);
  return mod;
}

auto RawPCMModulation::Build(const Configuration config) -> void {
  const auto mod_sf = static_cast<int32_t>(config.mod_sampling_freq());
  if (this->_sampling_freq < std::numeric_limits<Float>::epsilon()) this->_sampling_freq = static_cast<Float>(mod_sf);

  // up sampling
  std::vector<int32_t> sample_buf;
  const auto freq_ratio = static_cast<Float>(mod_sf) / _sampling_freq;
  sample_buf.resize(this->_buf.size() * static_cast<size_t>(freq_ratio));
  for (size_t i = 0; i < sample_buf.size(); i++) {
    const auto v = static_cast<Float>(i) / freq_ratio;
    const auto tmp = std::fmod(v, Float{1}) < 1 / freq_ratio ? this->_buf.at(static_cast<int>(v)) : 0;
    sample_buf.at(i) = tmp;
  }

  // LPF
  const auto num_tap = 31;
  const auto cutoff = _sampling_freq / 2 / static_cast<Float>(mod_sf);
  std::vector<Float> lpf(num_tap);
  for (auto i = 0; i < num_tap; i++) {
    const auto t = i - num_tap / 2;
    lpf.at(i) = Sinc(static_cast<Float>(t) * cutoff * 2);
  }

  auto max_v = std::numeric_limits<Float>::min();
  auto min_v = std::numeric_limits<Float>::max();
  std::vector<Float> lpf_buf;
  lpf_buf.resize(sample_buf.size(), 0);
  for (size_t i = 0; i < lpf_buf.size(); i++) {
    for (auto j = 0; j < num_tap; j++) {
      lpf_buf.at(i) += static_cast<Float>(sample_buf.at((i - j + sample_buf.size()) % sample_buf.size())) * lpf.at(j);
    }
    max_v = std::max<Float>(lpf_buf.at(i), max_v);
    min_v = std::min<Float>(lpf_buf.at(i), min_v);
  }

  if (max_v - min_v < std::numeric_limits<Float>::epsilon()) max_v = min_v + 1;
  this->buffer.resize(lpf_buf.size(), 0);
  for (size_t i = 0; i < lpf_buf.size(); i++) {
    this->buffer.at(i) = static_cast<uint8_t>(round(255 * ((lpf_buf.at(i) - min_v) / (max_v - min_v))));
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
      tmp.emplace_back(d);
    } else if (bits_per_sample == 16) {
      const auto d32 = static_cast<int32_t>(ReadFromStream<int16_t>(fs)) - std::numeric_limits<int16_t>::min();
      auto d8 = static_cast<uint8_t>(static_cast<float>(d32) / std::numeric_limits<uint16_t>::max() * std::numeric_limits<uint8_t>::max());
      tmp.emplace_back(d8);
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
  const auto freq_ratio = mod_sf / static_cast<double>(_sampling_freq);
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