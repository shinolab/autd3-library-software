// File: arrayfire_backend.hpp
// Project: gain
// Created Date: 10/09/2021
// Author: Shun Suzuki
// -----
// Last Modified: 22/11/2021
// Modified By: Shun Suzuki (suzuki@hapis.k.u-tokyo.ac.jp)
// -----
// Copyright (c) 2021 Hapis Lab. All rights reserved.
//

#pragma once

#include <vector>

#include "autd3/gain/backend.hpp"

namespace autd::gain::holo {

/**
 * \brief ArrayFire Backend for HoloGain (not optimized yet)
 */
class ArrayFireBackend : virtual public Backend {
 public:
  ArrayFireBackend() = default;
  ~ArrayFireBackend() override = default;
  ArrayFireBackend(const ArrayFireBackend& v) noexcept = default;
  ArrayFireBackend& operator=(const ArrayFireBackend& obj) = default;
  ArrayFireBackend(ArrayFireBackend&& obj) = default;
  ArrayFireBackend& operator=(ArrayFireBackend&& obj) = default;

  static BackendPtr create(int device_idx = 0);

  void sdp(const core::Geometry& geometry, const std::vector<core::Vector3>& foci, const std::vector<complex>& amps, double alpha, double lambda,
           size_t repeat, bool normalize, std::vector<core::Drive>& dst) override = 0;
  void evd(const core::Geometry& geometry, const std::vector<core::Vector3>& foci, const std::vector<complex>& amps, double gamma, bool normalize,
           std::vector<core::Drive>& dst) override = 0;
  void naive(const core::Geometry& geometry, const std::vector<core::Vector3>& foci, const std::vector<complex>& amps,
             std::vector<core::Drive>& dst) override = 0;
  void gs(const core::Geometry& geometry, const std::vector<core::Vector3>& foci, const std::vector<complex>& amps, size_t repeat,
          std::vector<core::Drive>& dst) override = 0;
  void gspat(const core::Geometry& geometry, const std::vector<core::Vector3>& foci, const std::vector<complex>& amps, size_t repeat,
             std::vector<core::Drive>& dst) override = 0;
  void lm(const core::Geometry& geometry, const std::vector<core::Vector3>& foci, const std::vector<complex>& amps, double eps_1, double eps_2,
          double tau, size_t k_max, const std::vector<double>& initial, std::vector<core::Drive>& dst) override = 0;
  void gauss_newton(const core::Geometry& geometry, const std::vector<core::Vector3>& foci, const std::vector<complex>& amps, double eps_1,
                    double eps_2, size_t k_max, const std::vector<double>& initial, std::vector<core::Drive>& dst) override = 0;
  void gradient_descent(const core::Geometry& geometry, const std::vector<core::Vector3>& foci, const std::vector<complex>& amps, double eps,
                        double step, size_t k_max, const std::vector<double>& initial, std::vector<core::Drive>& dst) override = 0;
  void apo(const core::Geometry& geometry, const std::vector<core::Vector3>& foci, const std::vector<complex>& amps, double eps, double lambda,
           size_t k_max, std::vector<core::Drive>& dst) override = 0;
  void greedy(const core::Geometry& geometry, const std::vector<core::Vector3>& foci, const std::vector<complex>& amps, size_t phase_div,
              std::vector<core::Drive>& dst) override = 0;
};

}  // namespace autd::gain::holo