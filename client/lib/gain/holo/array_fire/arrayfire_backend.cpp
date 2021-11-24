// File: arrayfire_backend.cpp
// Project: array_fire
// Created Date: 08/09/2021
// Author: Shun Suzuki
// -----
// Last Modified: 22/11/2021
// Modified By: Shun Suzuki (suzuki@hapis.k.u-tokyo.ac.jp)
// -----
// Copyright (c) 2021 Hapis Lab. All rights reserved.
//

#include "autd3/gain/arrayfire_backend.hpp"

#include "arrayfire_matrix.hpp"
#include "autd3/core/geometry.hpp"
#include "holo_impl.hpp"
#include "matrix_pool.hpp"

namespace autd::gain::holo {
class ArrayFireBackendImpl final : public ArrayFireBackend {
 public:
  ArrayFireBackendImpl() = default;
  ~ArrayFireBackendImpl() override = default;
  ArrayFireBackendImpl(const ArrayFireBackendImpl& v) noexcept = delete;
  ArrayFireBackendImpl& operator=(const ArrayFireBackendImpl& obj) = delete;
  ArrayFireBackendImpl(ArrayFireBackendImpl&& obj) = delete;
  ArrayFireBackendImpl& operator=(ArrayFireBackendImpl&& obj) = delete;

  void sdp(const core::Geometry& geometry, const std::vector<core::Vector3>& foci, const std::vector<complex>& amps, const double alpha,
           const double lambda, const size_t repeat, const bool normalize, std::vector<core::Drive>& dst) override {
    sdp_impl(_pool, geometry, foci, amps, alpha, lambda, repeat, normalize, dst);
  }
  void evd(const core::Geometry& geometry, const std::vector<core::Vector3>& foci, const std::vector<complex>& amps, const double gamma,
           const bool normalize, std::vector<core::Drive>& dst) override {
    evd_impl(_pool, geometry, foci, amps, gamma, normalize, dst);
  }
  void naive(const core::Geometry& geometry, const std::vector<core::Vector3>& foci, const std::vector<complex>& amps,
             std::vector<core::Drive>& dst) override {
    naive_impl(_pool, geometry, foci, amps, dst);
  }
  void gs(const core::Geometry& geometry, const std::vector<core::Vector3>& foci, const std::vector<complex>& amps, const size_t repeat,
          std::vector<core::Drive>& dst) override {
    gs_impl(_pool, geometry, foci, amps, repeat, dst);
  }
  void gspat(const core::Geometry& geometry, const std::vector<core::Vector3>& foci, const std::vector<complex>& amps, const size_t repeat,
             std::vector<core::Drive>& dst) override {
    gspat_impl(_pool, geometry, foci, amps, repeat, dst);
  }
  void lm(const core::Geometry& geometry, const std::vector<core::Vector3>& foci, const std::vector<complex>& amps, const double eps_1,
          const double eps_2, const double tau, const size_t k_max, const std::vector<double>& initial, std::vector<core::Drive>& dst) override {
    lm_impl(_pool, geometry, foci, amps, eps_1, eps_2, tau, k_max, initial, dst);
  }
  void gauss_newton(const core::Geometry& geometry, const std::vector<core::Vector3>& foci, const std::vector<complex>& amps, const double eps_1,
                    const double eps_2, const size_t k_max, const std::vector<double>& initial, std::vector<core::Drive>& dst) override {
    gauss_newton_impl(_pool, geometry, foci, amps, eps_1, eps_2, k_max, initial, dst);
  }
  void gradient_descent(const core::Geometry& geometry, const std::vector<core::Vector3>& foci, const std::vector<complex>& amps, const double eps,
                        const double step, const size_t k_max, const std::vector<double>& initial, std::vector<core::Drive>& dst) override {
    gradient_descent_impl(_pool, geometry, foci, amps, eps, step, k_max, initial, dst);
  }
  void apo(const core::Geometry& geometry, const std::vector<core::Vector3>& foci, const std::vector<complex>& amps, const double eps,
           const double lambda, const size_t k_max, std::vector<core::Drive>& dst) override {
    apo_impl(_pool, geometry, foci, amps, eps, lambda, 100, k_max, dst);
  }
  void greedy(const core::Geometry& geometry, const std::vector<core::Vector3>& foci, const std::vector<complex>& amps, const size_t phase_div,
              std::vector<core::Drive>& dst) override {
    greedy_impl(_pool, geometry, foci, amps, phase_div, dst);
  }

 private:
  MatrixBufferPool<AFMatrix<double>, AFMatrix<complex>> _pool;
};

BackendPtr ArrayFireBackend::create(const int device_idx) {
  af::setDevice(device_idx);
  return std::make_unique<ArrayFireBackendImpl>();
}

}  // namespace autd::gain::holo
