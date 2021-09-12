// File: c_api.cpp
// Project: holo_gain
// Created Date: 17/05/2021
// Author: Shun Suzuki
// -----
// Last Modified: 10/09/2021
// Modified By: Shun Suzuki (suzuki@hapis.k.u-tokyo.ac.jp)
// -----
// Copyright (c) 2021 Hapis Lab. All rights reserved.
//

#include "../base/wrapper.hpp"
#include "./holo_gain.h"
#include "autd3/gain/eigen_backend.hpp"
#include "autd3/gain/holo.hpp"

using Backend = autd::gain::holo::EigenBackend;

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
  auto* b = BackendCreate(autd::gain::holo::EigenBackend::create());
  *out = b;
}

void AUTDDeleteBackend(const void* backend) {
  const auto b = static_cast<const BackendWrapper*>(backend);
  BackendDelete(b);
}

void AUTDGainHoloSDP(void** gain, const void* backend, const double* points, const double* amps, const int32_t size, const double alpha,
                     const double lambda, const uint64_t repeat, const bool normalize) {
  const auto b = static_cast<const BackendWrapper*>(backend);
  const auto holo = PackFoci(points, size);
  const auto amps_ = PackAmps(amps, size);
  auto* g = GainCreate(autd::gain::holo::SDP::create(b->ptr, holo, amps_, alpha, lambda, repeat, normalize));
  *gain = g;
}

void AUTDGainHoloEVD(void** gain, const void* backend, const double* points, const double* amps, const int32_t size, const double gamma,
                     const bool normalize) {
  const auto b = static_cast<const BackendWrapper*>(backend);
  const auto holo = PackFoci(points, size);
  const auto amps_ = PackAmps(amps, size);
  auto* g = GainCreate(autd::gain::holo::EVD::create(b->ptr, holo, amps_, gamma, normalize));
  *gain = g;
}

void AUTDGainHoloNaive(void** gain, const void* backend, const double* points, const double* amps, const int32_t size) {
  const auto b = static_cast<const BackendWrapper*>(backend);
  const auto holo = PackFoci(points, size);
  const auto amps_ = PackAmps(amps, size);
  auto* g = GainCreate(autd::gain::holo::Naive::create(b->ptr, holo, amps_));
  *gain = g;
}

void AUTDGainHoloGS(void** gain, const void* backend, const double* points, const double* amps, const int32_t size, const uint64_t repeat) {
  const auto b = static_cast<const BackendWrapper*>(backend);
  const auto holo = PackFoci(points, size);
  const auto amps_ = PackAmps(amps, size);
  auto* g = GainCreate(autd::gain::holo::GS::create(b->ptr, holo, amps_, repeat));
  *gain = g;
}

void AUTDGainHoloGSPAT(void** gain, const void* backend, const double* points, const double* amps, const int32_t size, const uint64_t repeat) {
  const auto b = static_cast<const BackendWrapper*>(backend);
  const auto holo = PackFoci(points, size);
  const auto amps_ = PackAmps(amps, size);
  auto* g = GainCreate(autd::gain::holo::GSPAT::create(b->ptr, holo, amps_, repeat));
  *gain = g;
}

void AUTDGainHoloLM(void** gain, const void* backend, const double* points, const double* amps, const int32_t size, const double eps_1,
                    const double eps_2, const double tau, const uint64_t k_max, const double* initial, const int32_t initial_size) {
  const auto b = static_cast<const BackendWrapper*>(backend);
  const auto holo = PackFoci(points, size);
  const auto amps_ = PackAmps(amps, size);

  std::vector<double> initial_;
  initial_.reserve(initial_size);
  for (auto i = 0; i < initial_size; i++) initial_.emplace_back(initial[i]);

  auto* g = GainCreate(autd::gain::holo::LM::create(b->ptr, holo, amps_, eps_1, eps_2, tau, k_max, initial_));
  *gain = g;
}

void AUTDGainHoloGaussNewton(void** gain, const void* backend, const double* points, const double* amps, const int32_t size, const double eps_1,
                             const double eps_2, const uint64_t k_max, const double* initial, const int32_t initial_size) {
  const auto b = static_cast<const BackendWrapper*>(backend);
  const auto holo = PackFoci(points, size);
  const auto amps_ = PackAmps(amps, size);

  std::vector<double> initial_;
  initial_.reserve(initial_size);
  for (auto i = 0; i < initial_size; i++) initial_.emplace_back(initial[i]);

  auto* g = GainCreate(autd::gain::holo::GaussNewton::create(b->ptr, holo, amps_, eps_1, eps_2, k_max, initial_));
  *gain = g;
}
void AUTDGainHoloGradientDescent(void** gain, const void* backend, const double* points, const double* amps, const int32_t size, const double eps,
                                 const double step, const uint64_t k_max, const double* initial, const int32_t initial_size) {
  const auto b = static_cast<const BackendWrapper*>(backend);
  const auto holo = PackFoci(points, size);
  const auto amps_ = PackAmps(amps, size);

  std::vector<double> initial_;
  initial_.reserve(initial_size);
  for (auto i = 0; i < initial_size; i++) initial_.emplace_back(initial[i]);

  auto* g = GainCreate(autd::gain::holo::GradientDescent::create(b->ptr, holo, amps_, eps, step, k_max, initial_));
  *gain = g;
}
void AUTDGainHoloAPO(void** gain, const void* backend, const double* points, const double* amps, const int32_t size, const double eps,
                     const double lambda, const uint64_t k_max) {
  const auto b = static_cast<const BackendWrapper*>(backend);
  const auto holo = PackFoci(points, size);
  const auto amps_ = PackAmps(amps, size);

  auto* g = GainCreate(autd::gain::holo::APO::create(b->ptr, holo, amps_, eps, lambda, k_max));
  *gain = g;
}

void AUTDGainHoloGreedy(void** gain, const void* backend, const double* points, const double* amps, const int32_t size, const int32_t phase_div) {
  const auto b = static_cast<const BackendWrapper*>(backend);
  const auto holo = PackFoci(points, size);
  const auto amps_ = PackAmps(amps, size);

  auto* g = GainCreate(autd::gain::holo::Greedy::create(b->ptr, holo, amps_, static_cast<size_t>(phase_div)));
  *gain = g;
}
