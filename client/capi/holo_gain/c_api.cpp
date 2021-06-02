// File: c_api.cpp
// Project: holo_gain
// Created Date: 17/05/2021
// Author: Shun Suzuki
// -----
// Last Modified: 02/06/2021
// Modified By: Shun Suzuki (suzuki@hapis.k.u-tokyo.ac.jp)
// -----
// Copyright (c) 2021 Hapis Lab. All rights reserved.
//

#include "../base/wrapper.hpp"
#include "./holo_gain.h"
#include "eigen_backend.hpp"
#include "holo_gain.hpp"

namespace {
std::vector<autd::core::Vector3> PackFoci(const double* const points, const int32_t size) {
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

std::vector<double> PackAmps(const double* const amps, const int32_t size) {
  std::vector<double> amps_;
  amps_.reserve(size);
  for (auto i = 0; i < size; i++) amps_.emplace_back(static_cast<double>(amps[i]));
  return amps_;
}
}  // namespace

void AUTDEigen3Backend(void** out) {
  auto* b = BackendCreate(autd::gain::holo::Eigen3Backend::create());
  *out = b;
}

void AUTDDeleteBackend(void* backend) {
  const auto b = static_cast<BackendWrapper*>(backend);
  BackendDelete(b);
}

void AUTDHoloGainSDP(void** gain, void* backend, double* points, double* amps, const int32_t size, const double alpha, const double lambda,
                     const uint64_t repeat, const bool normalize) {
  auto holo = PackFoci(points, size);
  auto amps_ = PackAmps(amps, size);
  const auto b = static_cast<BackendWrapper*>(backend);
  auto* g = GainCreate(autd::gain::holo::HoloGainSDP::create(b->ptr, holo, amps_, alpha, lambda, repeat, normalize));
  *gain = g;
}

void AUTDHoloGainEVD(void** gain, void* backend, double* points, double* amps, const int32_t size, const double gamma, const bool normalize) {
  auto holo = PackFoci(points, size);
  auto amps_ = PackAmps(amps, size);
  const auto b = static_cast<BackendWrapper*>(backend);
  auto* g = GainCreate(autd::gain::holo::HoloGainEVD::create(b->ptr, holo, amps_, gamma, normalize));
  *gain = g;
}

void AUTDHoloGainNaive(void** gain, void* backend, double* points, double* amps, const int32_t size) {
  auto holo = PackFoci(points, size);
  auto amps_ = PackAmps(amps, size);
  const auto b = static_cast<BackendWrapper*>(backend);
  auto* g = GainCreate(autd::gain::holo::HoloGainNaive::create(b->ptr, holo, amps_));
  *gain = g;
}

void AUTDHoloGainGS(void** gain, void* backend, double* points, double* amps, const int32_t size, const uint64_t repeat) {
  auto holo = PackFoci(points, size);
  auto amps_ = PackAmps(amps, size);
  const auto b = static_cast<BackendWrapper*>(backend);
  auto* g = GainCreate(autd::gain::holo::HoloGainGS::create(b->ptr, holo, amps_, repeat));
  *gain = g;
}

void AUTDHoloGainGSPAT(void** gain, void* backend, double* points, double* amps, const int32_t size, const uint64_t repeat) {
  auto holo = PackFoci(points, size);
  auto amps_ = PackAmps(amps, size);
  const auto b = static_cast<BackendWrapper*>(backend);
  auto* g = GainCreate(autd::gain::holo::HoloGainGSPAT::create(b->ptr, holo, amps_, repeat));
  *gain = g;
}

void AUTDHoloGainLM(void** gain, void* backend, double* points, double* amps, const int32_t size, const double eps_1, const double eps_2,
                    const double tau, const uint64_t k_max, double* initial, const int32_t initial_size) {
  auto holo = PackFoci(points, size);
  auto amps_ = PackAmps(amps, size);

  std::vector<double> initial_;
  initial_.reserve(initial_size);
  for (auto i = 0; i < initial_size; i++) initial_.emplace_back(initial[i]);

  const auto b = static_cast<BackendWrapper*>(backend);
  auto* g = GainCreate(autd::gain::holo::HoloGainLM::create(b->ptr, holo, amps_, eps_1, eps_2, tau, k_max, initial_));
  *gain = g;
}

void AUTDHoloGainGreedy(void** gain, double* points, double* amps, const int32_t size, const int32_t phase_div) {
  auto holo = PackFoci(points, size);
  auto amps_ = PackAmps(amps, size);

  auto* g = GainCreate(autd::gain::holo::HoloGainGreedy::create(holo, amps_, static_cast<size_t>(phase_div)));
  *gain = g;
}
