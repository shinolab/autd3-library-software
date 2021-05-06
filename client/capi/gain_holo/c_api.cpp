// File: c_api.cpp
// Project: capi
// Created Date: 02/07/2018
// Author: Shun Suzuki
// -----
// Last Modified: 06/05/2021
// Modified By: Shun Suzuki (suzuki@hapis.k.u-tokyo.ac.jp)
// -----
// Copyright (c) 2018-2020 Hapis Lab. All rights reserved.
//

#include "../base/wrapper_gain.hpp"
#include "./gain_holo.h"
#include "gain/holo.hpp"

void AUTDHoloGainSDP(void** gain, float* points, float* amps, const int32_t size, const float alpha, const float lambda, const uint64_t repeat,
                     const bool normalize) {
  std::vector<autd::Vector3> holo;
  std::vector<autd::Float> amps_;
  for (auto i = 0; i < size; i++) {
    const auto x = static_cast<autd::Float>(points[3 * i]);
    const auto y = static_cast<autd::Float>(points[3 * i + 1]);
    const auto z = static_cast<autd::Float>(points[3 * i + 2]);
    holo.emplace_back(autd::Vector3(x, y, z));
    amps_.emplace_back(static_cast<autd::Float>(amps[i]));
  }
  auto* g = GainCreate(autd::gain::holo::HoloGainSDP<autd::gain::holo::Eigen3Backend>::Create(holo, amps_, static_cast<autd::Float>(alpha),
                                                                                              static_cast<autd::Float>(lambda), repeat, normalize));
  *gain = g;
}

void AUTDHoloGainEVD(void** gain, float* points, float* amps, const int32_t size, const float gamma, const bool normalize) {
  std::vector<autd::Vector3> holo;
  std::vector<autd::Float> amps_;
  for (auto i = 0; i < size; i++) {
    const auto x = static_cast<autd::Float>(points[3 * i]);
    const auto y = static_cast<autd::Float>(points[3 * i + 1]);
    const auto z = static_cast<autd::Float>(points[3 * i + 2]);
    holo.emplace_back(autd::Vector3(x, y, z));
    amps_.emplace_back(static_cast<autd::Float>(amps[i]));
  }
  auto* g =
      GainCreate(autd::gain::holo::HoloGainEVD<autd::gain::holo::Eigen3Backend>::Create(holo, amps_, static_cast<autd::Float>(gamma), normalize));
  *gain = g;
}

void AUTDHoloGainNaive(void** gain, float* points, float* amps, const int32_t size) {
  std::vector<autd::Vector3> holo;
  std::vector<autd::Float> amps_;
  for (auto i = 0; i < size; i++) {
    const auto x = static_cast<autd::Float>(points[3 * i]);
    const auto y = static_cast<autd::Float>(points[3 * i + 1]);
    const auto z = static_cast<autd::Float>(points[3 * i + 2]);
    holo.emplace_back(autd::Vector3(x, y, z));
    amps_.emplace_back(static_cast<autd::Float>(amps[i]));
  }
  auto* g = GainCreate(autd::gain::holo::HoloGainNaive<autd::gain::holo::Eigen3Backend>::Create(holo, amps_));
  *gain = g;
}

void AUTDHoloGainGS(void** gain, float* points, float* amps, const int32_t size, const uint64_t repeat) {
  std::vector<autd::Vector3> holo;
  std::vector<autd::Float> amps_;
  for (auto i = 0; i < size; i++) {
    const auto x = static_cast<autd::Float>(points[3 * i]);
    const auto y = static_cast<autd::Float>(points[3 * i + 1]);
    const auto z = static_cast<autd::Float>(points[3 * i + 2]);
    holo.emplace_back(autd::Vector3(x, y, z));
    amps_.emplace_back(static_cast<autd::Float>(amps[i]));
  }
  auto* g = GainCreate(autd::gain::holo::HoloGainGS<autd::gain::holo::Eigen3Backend>::Create(holo, amps_, repeat));
  *gain = g;
}

void AUTDHoloGainGSPAT(void** gain, float* points, float* amps, const int32_t size, const uint64_t repeat) {
  std::vector<autd::Vector3> holo;
  std::vector<autd::Float> amps_;
  for (auto i = 0; i < size; i++) {
    const auto x = static_cast<autd::Float>(points[3 * i]);
    const auto y = static_cast<autd::Float>(points[3 * i + 1]);
    const auto z = static_cast<autd::Float>(points[3 * i + 2]);
    holo.emplace_back(autd::Vector3(x, y, z));
    amps_.emplace_back(static_cast<autd::Float>(amps[i]));
  }
  auto* g = GainCreate(autd::gain::holo::HoloGainGSPAT<autd::gain::holo::Eigen3Backend>::Create(holo, amps_, repeat));
  *gain = g;
}

void AUTDHoloGainLM(void** gain, float* points, float* amps, const int32_t size, const float eps_1, const float eps_2, const float tau,
                    const uint64_t k_max, float* initial, const int32_t initial_size) {
  std::vector<autd::Vector3> holo;
  std::vector<autd::Float> amps_;
  for (auto i = 0; i < size; i++) {
    const auto x = static_cast<autd::Float>(points[3 * i]);
    const auto y = static_cast<autd::Float>(points[3 * i + 1]);
    const auto z = static_cast<autd::Float>(points[3 * i + 2]);
    holo.emplace_back(autd::Vector3(x, y, z));
    amps_.emplace_back(static_cast<autd::Float>(amps[i]));
  }
  std::vector<autd::Float> initial_;
  initial_.reserve(initial_size);
  for (auto i = 0; i < initial_size; i++) initial_.emplace_back(static_cast<autd::Float>(initial[i]));

  auto* g = GainCreate(autd::gain::holo::HoloGainLM<autd::gain::holo::Eigen3Backend>::Create(
      holo, amps_, static_cast<autd::Float>(eps_1), static_cast<autd::Float>(eps_2), static_cast<autd::Float>(tau), k_max, initial_));
  *gain = g;
}
