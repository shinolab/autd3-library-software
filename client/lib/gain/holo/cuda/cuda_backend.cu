// File: cuda_backend.cpp
// Project: cuda
// Created Date: 04/09/2021
// Author: Shun Suzuki
// -----
// Last Modified: 06/09/2021
// Modified By: Shun Suzuki (suzuki@hapis.k.u-tokyo.ac.jp)
// -----
// Copyright (c) 2021 Hapis Lab. All rights reserved.
//

#include <cublas_v2.h>
#include <thrust/device_vector.h>
#include <thrust/fill.h>

#include "./kernel.h"
#include "autd3/gain/cuda_backend.hpp"

namespace {
cublasOperation_t convert(autd::gain::holo::TRANSPOSE trans) {
  switch (trans) {
    case autd::gain::holo::TRANSPOSE::NO_TRANS:
      return cublasOperation_t::CUBLAS_OP_N;
    case autd::gain::holo::TRANSPOSE::CONJ_TRANS:
      return cublasOperation_t::CUBLAS_OP_C;
    case autd::gain::holo::TRANSPOSE::TRANS:
      return cublasOperation_t::CUBLAS_OP_T;
  }
  return cublasOperation_t::CUBLAS_OP_N;
}
}  // namespace

namespace autd {
namespace gain {
namespace holo {

template <typename T>
struct CuMatrix final : public Matrix<T> {
  explicit CuMatrix(const Eigen::Index row, const Eigen::Index col) : Matrix<T>(row, col), _row(row), _col(col), _d_vec(_row * _col) {}
  ~CuMatrix() override {}
  CuMatrix(const CuMatrix& obj) = delete;
  CuMatrix& operator=(const CuMatrix& obj) = delete;
  CuMatrix(const CuMatrix&& v) = delete;
  CuMatrix& operator=(CuMatrix&& obj) = delete;

  const T* ptr() const override { return _d_vec.data().get(); }
  T* ptr() override { return _d_vec.data().get(); }

  void set(const Eigen::Index row, const Eigen::Index col, T v) override { data(row, col) = v; }
  void get_col(const Eigen::Index i, std::shared_ptr<Matrix<T>> dst) override {
    const auto& col = data.col(i);
    std::memcpy(dst->data.data(), col.data(), sizeof(T) * col.size());
  }
  void fill(T v) override { thrust::fill(_d_vec.begin(), _d_vec.end(), v); }
  std::vector<T> get_diagonal() override {
    std::vector<T> v;
    // todo
    return v;
  }
  void set_diagonal(std::shared_ptr<Matrix<T>> v) override {  // todo
  }
  void set_diagonal(const T v) override {
    // todo
  }
  void copy_from(const std::vector<T>& v) override { cudaMemcpy(_d_vec.data().get(), v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice); }
  void copy_from(const T* v) override { cudaMemcpy(_d_vec.data().get(), v, _d_vec.size() * sizeof(T), cudaMemcpyHostToDevice); }
  void copy_to_host() override { cudaMemcpy(data.data(), _d_vec.data().get(), _row * _col * sizeof(T), cudaMemcpyDeviceToHost); }

 private:
  Eigen::Index _row;
  Eigen::Index _col;
  thrust::device_vector<T> _d_vec;
};

CUDABackend::CUDABackend() { cublasCreate_v2(&_handle); }
CUDABackend::~CUDABackend() { cublasDestroy_v2(_handle); }

template <typename T, typename C>
static std::shared_ptr<T> allocate_cu_matrix_impl(const std::string& name, const int64_t row, const int64_t col,
                                                  std::unordered_map<std::string, std::shared_ptr<T>>& cache) {
  const auto it = cache.find(name);
  if (it != cache.end()) {
    if (it->second->data.rows() == row && it->second->data.cols() == col) return it->second;
    cache.erase(name);
  }
  auto v = std::make_shared<C>(row, col);
  cache.emplace(name, v);
  return v;
}

std::shared_ptr<MatrixX> CUDABackend::allocate_matrix(const std::string& name, const size_t row, const size_t col) {
  return allocate_cu_matrix_impl<MatrixX, CuMatrix<double>>(name, row, col, _cache_mat);
}

std::shared_ptr<MatrixXc> CUDABackend::allocate_matrix_c(const std::string& name, const size_t row, const size_t col) {
  return allocate_cu_matrix_impl<MatrixXc, CuMatrix<complex>>(name, row, col, _cache_mat_c);
}

BackendPtr CUDABackend::create() { return std::make_shared<CUDABackend>(); }

void CUDABackend::make_complex(const std::shared_ptr<MatrixX> r, const std::shared_ptr<MatrixX> i, const std::shared_ptr<MatrixXc> c) {}
void CUDABackend::exp(const std::shared_ptr<MatrixXc> a) {}
void CUDABackend::scale(const std::shared_ptr<MatrixXc> a, const complex s) {}
void CUDABackend::hadamard_product(const std::shared_ptr<MatrixXc> a, const std::shared_ptr<MatrixXc> b, const std::shared_ptr<MatrixXc> c) {}
void CUDABackend::real(const std::shared_ptr<MatrixXc> a, const std::shared_ptr<MatrixX> b) {}
void CUDABackend::arg(const std::shared_ptr<MatrixXc> a, const std::shared_ptr<MatrixXc> c) {}
void CUDABackend::pseudo_inverse_svd(const std::shared_ptr<MatrixXc> matrix, const double alpha, const std::shared_ptr<MatrixXc> result) {}
std::shared_ptr<MatrixXc> CUDABackend::max_eigen_vector(const std::shared_ptr<MatrixXc> matrix) { return std::make_shared<CuMatrix<complex>>(0, 0); }

void CUDABackend::matrix_add(const double alpha, const std::shared_ptr<MatrixX> a, const std::shared_ptr<MatrixX> b) {}
void CUDABackend::matrix_mul(const TRANSPOSE trans_a, const TRANSPOSE trans_b, const complex alpha, const std::shared_ptr<MatrixXc> a,
                             const std::shared_ptr<MatrixXc> b, const complex beta, const std::shared_ptr<MatrixXc> c) {
  const auto lda = static_cast<int>(a->data.rows());
  const auto ldb = static_cast<int>(b->data.rows());
  const auto ldc = trans_a == TRANSPOSE::NO_TRANS ? static_cast<int>(a->data.rows()) : static_cast<int>(a->data.cols());
  const auto n = trans_b == TRANSPOSE::NO_TRANS ? static_cast<int>(b->data.cols()) : static_cast<int>(b->data.rows());
  const auto k = trans_a == TRANSPOSE::NO_TRANS ? static_cast<int>(a->data.cols()) : static_cast<int>(a->data.rows());
  cublasZgemm3m(_handle, convert(trans_a), convert(trans_b), ldc, n, k, (const cuDoubleComplex*)&alpha, (const cuDoubleComplex*)a->ptr(), lda,
                (const cuDoubleComplex*)b->ptr(), ldb, (const cuDoubleComplex*)&beta, (cuDoubleComplex*)c->ptr(), ldc);
}
void CUDABackend::matrix_mul(const TRANSPOSE trans_a, const TRANSPOSE trans_b, const double alpha, const std::shared_ptr<MatrixX> a,
                             const std::shared_ptr<MatrixX> b, const double beta, const std::shared_ptr<MatrixX> c) {}

void CUDABackend::solve_g(const std::shared_ptr<MatrixX> a, const std::shared_ptr<MatrixX> b, const std::shared_ptr<MatrixX> c) {}
void CUDABackend::solve_ch(const std::shared_ptr<MatrixXc> a, const std::shared_ptr<MatrixXc> b) {}
double CUDABackend::dot(const std::shared_ptr<MatrixX> a, const std::shared_ptr<MatrixX> b) {
  double d;
  cublasDdot(_handle, static_cast<int>(a->data.size()), a->ptr(), 1, b->ptr(), 1, &d);
  return d;
}
complex CUDABackend::dot(const std::shared_ptr<MatrixXc> a, const std::shared_ptr<MatrixXc> b) {
  complex d;
  cublasZdotc(_handle, static_cast<int>(a->data.size()), (const cuDoubleComplex*)a->ptr(), 1, (const cuDoubleComplex*)b->ptr(), 1,
              (cuDoubleComplex*)&d);
  return d;
}
double CUDABackend::max_coefficient(const std::shared_ptr<MatrixXc> v) { return 0.0; }
double CUDABackend::max_coefficient(const std::shared_ptr<MatrixX> v) { return 0.0; }
std::shared_ptr<MatrixXc> CUDABackend::concat_row(const std::shared_ptr<MatrixXc> a, const std::shared_ptr<MatrixXc> b) {
  return std::make_shared<CuMatrix<complex>>(0, 0);
}
std::shared_ptr<MatrixXc> CUDABackend::concat_col(const std::shared_ptr<MatrixXc> a, const std::shared_ptr<MatrixXc> b) {
  return std::make_shared<CuMatrix<complex>>(0, 0);
}
void CUDABackend::mat_cpy(const std::shared_ptr<MatrixX> a, const std::shared_ptr<MatrixX> b) {
  cublasDcopy(_handle, static_cast<int>(a->data.rows()) * static_cast<int>(a->data.cols()), a->ptr(), 1, b->ptr(), 1);
}
void CUDABackend::mat_cpy(const std::shared_ptr<MatrixXc> a, const std::shared_ptr<MatrixXc> b) {
  cublasZcopy(_handle, static_cast<int>(a->data.rows()) * static_cast<int>(a->data.cols()), (const cuDoubleComplex*)a->ptr(), 1,
              (cuDoubleComplex*)b->ptr(), 1);
}

void CUDABackend::set_from_complex_drive(std::vector<core::DataArray>& data, const std::shared_ptr<MatrixXc> drive, const bool normalize,
                                         const double max_coefficient) {}

std::shared_ptr<MatrixXc> CUDABackend::transfer_matrix(const double* foci, size_t foci_num, const std::vector<const double*>& positions,
                                                       const std::vector<const double*>& directions, double wavelength, double attenuation) {
  const auto m = static_cast<Eigen::Index>(foci_num);
  const auto n = static_cast<Eigen::Index>(positions.size() * core::NUM_TRANS_IN_UNIT);

  auto g = allocate_matrix_c("g", m, n);

  // const auto wave_number = 2.0 * M_PI / geometry->wavelength();
  // const auto attenuation = geometry->attenuation_coefficient();
  // for (Eigen::Index i = 0; i < m; i++) {
  //    const auto& tp = foci[i];
  //    for (Eigen::Index j = 0; j < n; j++) {
  //        const auto& pos = geometry->position(j);
  //        const auto& dir = geometry->direction(j / core::NUM_TRANS_IN_UNIT);
  //        g->data(i, j) = utils::transfer(pos, dir, tp, wave_number, attenuation);
  //    }
  //}
  return g;
}

void CUDABackend::set_bcd_result(const std::shared_ptr<MatrixXc> mat, const std::shared_ptr<MatrixXc> vec, const size_t idx) {}

std::shared_ptr<MatrixXc> CUDABackend::back_prop(const std::shared_ptr<MatrixXc> transfer, const std::vector<complex>& amps) {
  const auto m = transfer->data.rows();
  const auto n = transfer->data.cols();

  // Eigen::Matrix<double, -1, 1, Eigen::ColMajor> denominator(m);
  // for (Eigen::Index i = 0; i < m; i++) {
  //    auto tmp = 0.0;
  //    for (Eigen::Index j = 0; j < n; j++) tmp += std::abs(transfer->data(i, j));
  //    denominator(i) = tmp;
  //}

  auto b = allocate_matrix_c("b", n, m);
  // for (Eigen::Index i = 0; i < m; i++) {
  //    auto c = amps[i] / denominator(i);
  //    for (Eigen::Index j = 0; j < n; j++) b->data(j, i) = c * std::conj(transfer->data(i, j));
  //}
  return b;
}

std::shared_ptr<MatrixXc> CUDABackend::sigma_regularization(const std::shared_ptr<MatrixXc> transfer, const std::vector<complex>& amps,
                                                            const double gamma) {
  const auto m = transfer->data.rows();
  const auto n = transfer->data.cols();

  auto sigma = allocate_matrix_c("sigma", n, n);
  sigma->fill(ZERO);
  // for (Eigen::Index j = 0; j < n; j++) {
  //    double tmp = 0;
  //    for (Eigen::Index i = 0; i < m; i++) tmp += std::abs(transfer->data(i, j) * amps[i]);
  //    sigma->data(j, j) = complex(std::pow(std::sqrt(tmp / static_cast<double>(m)), gamma), 0.0);
  //}
  return sigma;
}

void CUDABackend::col_sum_imag(const std::shared_ptr<MatrixXc> mat, const std::shared_ptr<MatrixX> dst) {}

}  // namespace holo
}  // namespace gain
}  // namespace autd
