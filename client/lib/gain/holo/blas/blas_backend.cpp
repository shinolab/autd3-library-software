// File: blas_backend.cpp
// Project: src
// Created Date: 17/05/2021
// Author: Shun Suzuki
// -----
// Last Modified: 10/09/2021
// Modified By: Shun Suzuki (suzuki@hapis.k.u-tokyo.ac.jp)
// -----
// Copyright (c) 2021 Hapis Lab. All rights reserved.
//

#include "autd3/gain/blas_backend.hpp"

#include "blas_matrix.hpp"
#include "holo_impl.hpp"
#include "matrix_pool.hpp"

namespace autd::gain::holo {
class BLASBackendImpl final : public BLASBackend {
 public:
  BLASBackendImpl() = default;
  ~BLASBackendImpl() override = default;
  BLASBackendImpl(const BLASBackendImpl& v) noexcept = delete;
  BLASBackendImpl& operator=(const BLASBackendImpl& obj) = delete;
  BLASBackendImpl(BLASBackendImpl&& obj) = delete;
  BLASBackendImpl& operator=(BLASBackendImpl&& obj) = delete;

  void sdp(const core::GeometryPtr& geometry, const std::vector<core::Vector3>& foci, const std::vector<complex>& amps, const double alpha,
           const double lambda, const size_t repeat, const bool normalize, std::vector<core::DataArray>& dst) override {
    sdp_impl(_pool, geometry, foci, amps, alpha, lambda, repeat, normalize, dst);
  }
  void evd(const core::GeometryPtr& geometry, const std::vector<core::Vector3>& foci, const std::vector<complex>& amps, const double gamma,
           const bool normalize, std::vector<core::DataArray>& dst) override {
    evd_impl(_pool, geometry, foci, amps, gamma, normalize, dst);
  }
  void naive(const core::GeometryPtr& geometry, const std::vector<core::Vector3>& foci, const std::vector<complex>& amps,
             std::vector<core::DataArray>& dst) override {
    naive_impl(_pool, geometry, foci, amps, dst);
  }
  void gs(const core::GeometryPtr& geometry, const std::vector<core::Vector3>& foci, const std::vector<complex>& amps, const size_t repeat,
          std::vector<core::DataArray>& dst) override {
    gs_impl(_pool, geometry, foci, amps, repeat, dst);
  }
  void gspat(const core::GeometryPtr& geometry, const std::vector<core::Vector3>& foci, const std::vector<complex>& amps, const size_t repeat,
             std::vector<core::DataArray>& dst) override {
    gspat_impl(_pool, geometry, foci, amps, repeat, dst);
  }
  void lm(const core::GeometryPtr& geometry, const std::vector<core::Vector3>& foci, const std::vector<complex>& amps, const double eps_1,
          const double eps_2, const double tau, const size_t k_max, const std::vector<double>& initial, std::vector<core::DataArray>& dst) override {
    lm_impl(_pool, geometry, foci, amps, eps_1, eps_2, tau, k_max, initial, dst);
  }
  void gauss_newton(const core::GeometryPtr& geometry, const std::vector<core::Vector3>& foci, const std::vector<complex>& amps, const double eps_1,
                    const double eps_2, const size_t k_max, const std::vector<double>& initial, std::vector<core::DataArray>& dst) override {
    gauss_newton_impl(_pool, geometry, foci, amps, eps_1, eps_2, k_max, initial, dst);
  }
  void gradient_descent(const core::GeometryPtr& geometry, const std::vector<core::Vector3>& foci, const std::vector<complex>& amps, const double eps,
                        const double step, const size_t k_max, const std::vector<double>& initial, std::vector<core::DataArray>& dst) override {
    gradient_descent_impl(_pool, geometry, foci, amps, eps, step, k_max, initial, dst);
  }
  void apo(const core::GeometryPtr& geometry, const std::vector<core::Vector3>& foci, const std::vector<complex>& amps, const double eps,
           const double lambda, const size_t k_max, std::vector<core::DataArray>& dst) override {
    apo_impl(_pool, geometry, foci, amps, eps, lambda, 100, k_max, dst);
  }
  void greedy(const core::GeometryPtr& geometry, const std::vector<core::Vector3>& foci, const std::vector<complex>& amps, const size_t phase_div,
              std::vector<core::DataArray>& dst) override {
    greedy_impl(geometry, foci, amps, phase_div, dst);
  }

 private:
  MatrixBufferPool<BLASMatrix<double>, BLASMatrix<complex>> _pool;
};

BackendPtr BLASBackend::create() { return std::make_unique<BLASBackendImpl>(); }

}  // namespace autd::gain::holo
