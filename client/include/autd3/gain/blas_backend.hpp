// File: eigen_backend.hpp
// Project: gain
// Created Date: 10/09/2021
// Author: Shun Suzuki
// -----
// Last Modified: 10/09/2021
// Modified By: Shun Suzuki (suzuki@hapis.k.u-tokyo.ac.jp)
// -----
// Copyright (c) 2021 Hapis Lab. All rights reserved.
//

#pragma once

#include "autd3/gain/backend.hpp"

namespace autd::gain::holo {

class BLASBackend : virtual public Backend {
 public:
  BLASBackend() = default;
  ~BLASBackend() override = default;
  BLASBackend(const BLASBackend& v) noexcept = default;
  BLASBackend& operator=(const BLASBackend& obj) = default;
  BLASBackend(BLASBackend&& obj) = default;
  BLASBackend& operator=(BLASBackend&& obj) = default;

  static BackendPtr create();

  void sdp(const core::GeometryPtr& geometry, const std::vector<core::Vector3>& foci, const std::vector<complex>& amps, double alpha, double lambda,
           size_t repeat, bool normalize, std::vector<core::DataArray>& dst) override = 0;
  void evd(const core::GeometryPtr& geometry, const std::vector<core::Vector3>& foci, const std::vector<complex>& amps, double gamma, bool normalize,
           std::vector<core::DataArray>& dst) override = 0;
  void naive(const core::GeometryPtr& geometry, const std::vector<core::Vector3>& foci, const std::vector<complex>& amps,
             std::vector<core::DataArray>& dst) override = 0;
  void gs(const core::GeometryPtr& geometry, const std::vector<core::Vector3>& foci, const std::vector<complex>& amps, size_t repeat,
          std::vector<core::DataArray>& dst) override = 0;
  void gspat(const core::GeometryPtr& geometry, const std::vector<core::Vector3>& foci, const std::vector<complex>& amps, size_t repeat,
             std::vector<core::DataArray>& dst) override = 0;
  void lm(const core::GeometryPtr& geometry, const std::vector<core::Vector3>& foci, const std::vector<complex>& amps, double eps_1, double eps_2,
          double tau, size_t k_max, const std::vector<double>& initial, std::vector<core::DataArray>& dst) override = 0;
  void gauss_newton(const core::GeometryPtr& geometry, const std::vector<core::Vector3>& foci, const std::vector<complex>& amps, double eps_1,
                    double eps_2, size_t k_max, const std::vector<double>& initial, std::vector<core::DataArray>& dst) override = 0;
  void gradient_descent(const core::GeometryPtr& geometry, const std::vector<core::Vector3>& foci, const std::vector<complex>& amps, double eps,
                        double step, size_t k_max, const std::vector<double>& initial, std::vector<core::DataArray>& dst) override = 0;
  void apo(const core::GeometryPtr& geometry, const std::vector<core::Vector3>& foci, const std::vector<complex>& amps, double eps, double lambda,
           size_t k_max, std::vector<core::DataArray>& dst) override = 0;
  void greedy(const core::GeometryPtr& geometry, const std::vector<core::Vector3>& foci, const std::vector<complex>& amps, size_t phase_div,
              std::vector<core::DataArray>& dst) override = 0;
};

}  // namespace autd::gain::holo
