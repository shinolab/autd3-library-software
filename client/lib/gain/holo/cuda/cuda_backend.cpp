// File: cuda_backend.cpp
// Project: cuda
// Created Date: 10/09/2021
// Author: Shun Suzuki
// -----
// Last Modified: 10/09/2021
// Modified By: Shun Suzuki (suzuki@hapis.k.u-tokyo.ac.jp)
// -----
// Copyright (c) 2021 Hapis Lab. All rights reserved.
//

#include "autd3/gain/cuda_backend.hpp"

#include "cuda_matrix.hpp"
#include "holo_impl.hpp"
#include "matrix_pool.hpp"

namespace autd::gain::holo {
class CUDABackendImpl : public CUDABackend {
 public:
  explicit CUDABackendImpl(const int device_idx) { CuContext::init(device_idx); }
  ~CUDABackendImpl() override { CuContext::free(); }
  CUDABackendImpl(const CUDABackendImpl& v) noexcept = delete;
  CUDABackendImpl& operator=(const CUDABackendImpl& obj) = delete;
  CUDABackendImpl(CUDABackendImpl&& obj) = delete;
  CUDABackendImpl& operator=(CUDABackendImpl&& obj) = delete;

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
  MatrixBufferPool<EigenMatrix<double>, EigenMatrix<complex>> _pool;
};

BackendPtr CUDABackend::create(const int device_idx) { return std::make_unique<CUDABackendImpl>(device_idx); }

}  // namespace autd::gain::holo
