// File: c_api.cpp
// Project: holo_gain
// Created Date: 17/05/2021
// Author: Shun Suzuki
// -----
// Last Modified: 17/05/2021
// Modified By: Shun Suzuki (suzuki@hapis.k.u-tokyo.ac.jp)
// -----
// Copyright (c) 2021 Hapis Lab. All rights reserved.
//

#include "../base/wrapper_gain.hpp"
#include "./holo_gain.h"
#include "eigen_backend.hpp"
#include "holo_gain.hpp"
#include "wrapper_backend.hpp"

namespace {
std::vector<autd::core::Vector3> pack_foci(const double* const points, const int32_t size) {
  std::vector<autd::core::Vector3> holo;
  holo.reserve(size);
  for (auto i = 0; i < size; i++) {
    const auto x = points[3 * i];
    const auto y = points[3 * i + 1];
    const auto z = points[3 * i + 2];
    holo.emplace_back(autd::core::Vector3(x, y, z));
  }
  return holo;
}

std::vector<double> pack_amps(const double* const amps, const int32_t size) {
  std::vector<double> amps_;
  amps_.reserve(size);
  for (auto i = 0; i < size; i++) amps_.emplace_back(static_cast<double>(amps[i]));
  return amps_;
}
}  // namespace

void AUTDEigen3Backend(void** out) {
  auto* b = BackendCreate(autd::gain::holo::Eigen3Backend::Create());
  *out = b;
}

void AUTDHoloGainSDP(void** gain, void* backend, double* points, double* amps, const int32_t size, const double alpha, const double lambda,
                     const uint64_t repeat, const bool normalize) {
  auto holo = pack_foci(points, size);
  auto amps_ = pack_amps(amps, size);
  const auto b = static_cast<BackendWrapper*>(backend);
  auto* g = GainCreate(autd::gain::holo::HoloGainSDP::Create(b->ptr, holo, amps_, alpha, lambda, repeat, normalize));
  *gain = g;
}

void AUTDHoloGainEVD(void** gain, void* backend, double* points, double* amps, const int32_t size, const double gamma, const bool normalize) {
  auto holo = pack_foci(points, size);
  auto amps_ = pack_amps(amps, size);
  const auto b = static_cast<BackendWrapper*>(backend);
  auto* g = GainCreate(autd::gain::holo::HoloGainEVD::Create(b->ptr, holo, amps_, gamma, normalize));
  *gain = g;
}

void AUTDHoloGainNaive(void** gain, void* backend, double* points, double* amps, const int32_t size) {
  auto holo = pack_foci(points, size);
  auto amps_ = pack_amps(amps, size);
  const auto b = static_cast<BackendWrapper*>(backend);
  auto* g = GainCreate(autd::gain::holo::HoloGainNaive::Create(b->ptr, holo, amps_));
  *gain = g;
}

void AUTDHoloGainGS(void** gain, void* backend, double* points, double* amps, const int32_t size, const uint64_t repeat) {
  auto holo = pack_foci(points, size);
  auto amps_ = pack_amps(amps, size);
  const auto b = static_cast<BackendWrapper*>(backend);
  auto* g = GainCreate(autd::gain::holo::HoloGainGS::Create(b->ptr, holo, amps_, repeat));
  *gain = g;
}

void AUTDHoloGainGSPAT(void** gain, void* backend, double* points, double* amps, const int32_t size, const uint64_t repeat) {
  auto holo = pack_foci(points, size);
  auto amps_ = pack_amps(amps, size);
  const auto b = static_cast<BackendWrapper*>(backend);
  auto* g = GainCreate(autd::gain::holo::HoloGainGSPAT::Create(b->ptr, holo, amps_, repeat));
  *gain = g;
}

void AUTDHoloGainLM(void** gain, void* backend, double* points, double* amps, const int32_t size, const double eps_1, const double eps_2,
                    const double tau, const uint64_t k_max, double* initial, const int32_t initial_size) {
  auto holo = pack_foci(points, size);
  auto amps_ = pack_amps(amps, size);

  std::vector<double> initial_;
  initial_.reserve(initial_size);
  for (auto i = 0; i < initial_size; i++) initial_.emplace_back(initial[i]);

  const auto b = static_cast<BackendWrapper*>(backend);
  auto* g = GainCreate(autd::gain::holo::HoloGainLM::Create(b->ptr, holo, amps_, eps_1, eps_2, tau, k_max, initial_));
  *gain = g;
}
