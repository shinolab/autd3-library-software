// File: backend.hpp
// Project: gain
// Created Date: 10/09/2021
// Author: Shun Suzuki
// -----
// Last Modified: 13/12/2021
// Modified By: Shun Suzuki (suzuki@hapis.k.u-tokyo.ac.jp)
// -----
// Copyright (c) 2021 Hapis Lab. All rights reserved.
//

#pragma once

#include <memory>
#include <vector>

#include "autd3/core/geometry.hpp"
#include "matrix.hpp"

namespace autd {
namespace gain {
namespace holo {

/**
 * \brief Backend for HoloGain
 */
class Backend {
 public:
  Backend() = default;
  virtual ~Backend() = default;
  Backend(const Backend& v) noexcept = default;
  Backend& operator=(const Backend& obj) = default;
  Backend(Backend&& obj) = default;
  Backend& operator=(Backend&& obj) = default;

  virtual void sdp(const core::Geometry& geometry, const std::vector<core::Vector3>& foci, const std::vector<complex>& amps, double alpha,
                   double lambda, size_t repeat, bool normalize, std::vector<core::Drive>& dst) = 0;
  virtual void evd(const core::Geometry& geometry, const std::vector<core::Vector3>& foci, const std::vector<complex>& amps, double gamma,
                   bool normalize, std::vector<core::Drive>& dst) = 0;
  virtual void naive(const core::Geometry& geometry, const std::vector<core::Vector3>& foci, const std::vector<complex>& amps,
                     std::vector<core::Drive>& dst) = 0;
  virtual void gs(const core::Geometry& geometry, const std::vector<core::Vector3>& foci, const std::vector<complex>& amps, size_t repeat,
                  std::vector<core::Drive>& dst) = 0;
  virtual void gspat(const core::Geometry& geometry, const std::vector<core::Vector3>& foci, const std::vector<complex>& amps, size_t repeat,
                     std::vector<core::Drive>& dst) = 0;
  virtual void lm(const core::Geometry& geometry, const std::vector<core::Vector3>& foci, const std::vector<complex>& amps, double eps_1,
                  double eps_2, double tau, size_t k_max, const std::vector<double>& initial, std::vector<core::Drive>& dst) = 0;
  virtual void gauss_newton(const core::Geometry& geometry, const std::vector<core::Vector3>& foci, const std::vector<complex>& amps, double eps_1,
                            double eps_2, size_t k_max, const std::vector<double>& initial, std::vector<core::Drive>& dst) = 0;
  virtual void gradient_descent(const core::Geometry& geometry, const std::vector<core::Vector3>& foci, const std::vector<complex>& amps, double eps,
                                double step, size_t k_max, const std::vector<double>& initial, std::vector<core::Drive>& dst) = 0;
  virtual void apo(const core::Geometry& geometry, const std::vector<core::Vector3>& foci, const std::vector<complex>& amps, double eps,
                   double lambda, size_t k_max, std::vector<core::Drive>& dst) = 0;
  virtual void greedy(const core::Geometry& geometry, const std::vector<core::Vector3>& foci, const std::vector<complex>& amps, size_t phase_div,
                      std::vector<core::Drive>& dst) = 0;
};

using BackendPtr = std::shared_ptr<Backend>;

}  // namespace holo
}  // namespace gain
}  // namespace autd
