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

#include <thrust/device_vector.h>
#include <thrust/fill.h>
#include <thrust/reduce.h>

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

  [[nodiscard]] T at(size_t row, size_t col) const override {
    T r;
    cudaMemcpy(&r, (const void*)(&_d_vec[row + col * _row]).get(), sizeof(T), cudaMemcpyDeviceToHost);
    return r;
  };
  const T* ptr() const override { return _d_vec.data().get(); }
  T* ptr() override { return _d_vec.data().get(); }
  [[nodiscard]] double max_element() const override;
  void set(const Eigen::Index row, const Eigen::Index col, T v) override {
    cudaMemcpy((void*)(&_d_vec[row + col * _row]).get(), &v, sizeof(T), cudaMemcpyHostToDevice);
  }
  void get_col(const Eigen::Index i, std::shared_ptr<Matrix<T>> dst) override {
    cudaMemcpy(dst->ptr(), ptr() + i * _row, _row * sizeof(T), cudaMemcpyDeviceToDevice);
  }
  void fill(T v) override { thrust::fill(_d_vec.begin(), _d_vec.end(), v); }
  void get_diagonal(std::shared_ptr<Matrix<T>> v) override { cu_get_diagonal(ptr(), v->ptr(), (uint32_t)(std::min)(data.rows(), data.cols())); }
  void set_diagonal(std::shared_ptr<Matrix<T>> v) override { cu_set_diagonal(v->ptr(), ptr(), (uint32_t)(std::min)(data.rows(), data.cols())); }
  void copy_from(const std::vector<T>& v) override { cudaMemcpy(_d_vec.data().get(), v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice); }
  void copy_from(const T* v) override { cudaMemcpy(_d_vec.data().get(), v, _d_vec.size() * sizeof(T), cudaMemcpyHostToDevice); }
  void copy_to_host() override { cudaMemcpy(data.data(), _d_vec.data().get(), _row * _col * sizeof(T), cudaMemcpyDeviceToHost); }

 private:
  Eigen::Index _row;
  Eigen::Index _col;
  thrust::device_vector<T> _d_vec;
};

template <>
double CuMatrix<double>::max_element() const {
  return *thrust::max_element(_d_vec.begin(), _d_vec.end());
}

template <>
double CuMatrix<complex>::max_element() const {
  throw std::runtime_error("not impletemted max_element for complex");
}

CUDABackend::CUDABackend() {
  cublasCreate_v2(&_handle);
  cusolverDnCreate(&_handle_s);
}
CUDABackend::~CUDABackend() {
  cublasDestroy_v2(_handle);
  cusolverDnDestroy(_handle_s);
}

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

void CUDABackend::make_complex(const std::shared_ptr<MatrixX> r, const std::shared_ptr<MatrixX> i, const std::shared_ptr<MatrixXc> c) {
  cu_make_complex(r->ptr(), i->ptr(), (uint32_t)c->data.rows(), (uint32_t)c->data.cols(), (cuDoubleComplex*)c->ptr());
}
void CUDABackend::exp(const std::shared_ptr<MatrixXc> a) { cu_exp((uint32_t)a->data.rows(), (uint32_t)a->data.cols(), (cuDoubleComplex*)a->ptr()); }
void CUDABackend::scale(const std::shared_ptr<MatrixXc> a, const complex s) {
  cublasZscal_v2(_handle, static_cast<int>(a->data.size()), (const cuDoubleComplex*)&s, (cuDoubleComplex*)a->ptr(), 1);
}
void CUDABackend::hadamard_product(const std::shared_ptr<MatrixXc> a, const std::shared_ptr<MatrixXc> b, const std::shared_ptr<MatrixXc> c) {
  cu_hadamard_product((const cuDoubleComplex*)a->ptr(), (const cuDoubleComplex*)b->ptr(), (uint32_t)c->data.rows(), (uint32_t)c->data.cols(),
                      (cuDoubleComplex*)c->ptr());
}
void CUDABackend::real(const std::shared_ptr<MatrixXc> a, const std::shared_ptr<MatrixX> b) {
  cu_real((const cuDoubleComplex*)a->ptr(), (uint32_t)a->data.rows(), (uint32_t)a->data.cols(), b->ptr());
}
void CUDABackend::arg(const std::shared_ptr<MatrixXc> a, const std::shared_ptr<MatrixXc> c) {
  cu_arg((const cuDoubleComplex*)a->ptr(), (uint32_t)a->data.rows(), (uint32_t)a->data.cols(), (cuDoubleComplex*)c->ptr());
}
void CUDABackend::pseudo_inverse_svd(const std::shared_ptr<MatrixXc> matrix, const double alpha, const std::shared_ptr<MatrixXc> result) {
  const auto nc = matrix->data.cols();
  const auto nr = matrix->data.rows();

  const auto lda = static_cast<int>(nr);
  const auto ldu = static_cast<int>(nr);
  const auto ldv = static_cast<int>(nc);

  const auto s_size = std::min(nr, nc);
  double* d_s = nullptr;
  cudaMalloc((void**)&d_s, sizeof(double) * s_size);
  const auto a = this->allocate_matrix_c("_pis_a", matrix->data.rows(), matrix->data.cols());
  const auto u = this->allocate_matrix_c("_pis_u", nr, nr);
  const auto v = this->allocate_matrix_c("_pis_v", nc, nc);
  cudaMemcpy(a->ptr(), matrix->ptr(), matrix->data.rows() * matrix->data.cols() * sizeof(complex), cudaMemcpyDeviceToDevice);

  size_t workspace_in_bytes_on_device;
  size_t workspace_in_bytes_on_host;

  cusolverDnXgesvdp_bufferSize(_handle_s, NULL, cusolverEigMode_t::CUSOLVER_EIG_MODE_VECTOR, 0, static_cast<int>(nr), static_cast<int>(nc),
                               cudaDataType::CUDA_C_64F, a->ptr(), lda, cudaDataType::CUDA_R_64F, d_s, cudaDataType::CUDA_C_64F, u->ptr(), ldu,
                               cudaDataType::CUDA_C_64F, v->ptr(), ldv, cudaDataType::CUDA_C_64F, &workspace_in_bytes_on_device,
                               &workspace_in_bytes_on_host);
  void* workspace_buffer_on_device = nullptr;
  void* workspace_buffer_on_host = nullptr;
  cudaMalloc((void**)&workspace_buffer_on_device, workspace_in_bytes_on_device);
  if (workspace_in_bytes_on_host > 0) workspace_buffer_on_host = (void*)malloc(workspace_in_bytes_on_host);

  int* info;
  cudaMalloc((void**)&info, sizeof(int));
  double h_err_sigma;
  cusolverDnXgesvdp(_handle_s, NULL, cusolverEigMode_t::CUSOLVER_EIG_MODE_VECTOR, 0, static_cast<int>(nr), static_cast<int>(nc),
                    cudaDataType::CUDA_C_64F, a->ptr(), lda, cudaDataType::CUDA_R_64F, d_s, cudaDataType::CUDA_C_64F, u->ptr(), ldu,
                    cudaDataType::CUDA_C_64F, v->ptr(), ldv, cudaDataType::CUDA_C_64F, workspace_buffer_on_device, workspace_in_bytes_on_device,
                    workspace_buffer_on_host, workspace_in_bytes_on_host, info, &h_err_sigma);

  const auto singular_inv = this->allocate_matrix_c("_pis_si", nc, nr);
  calc_singular_inv(d_s, (uint32_t)s_size, alpha, (cuDoubleComplex*)singular_inv->ptr());

  const auto tmp = this->allocate_matrix_c("_pis_tmp", nc, nr);
  CUDABackend::matrix_mul(TRANSPOSE::NO_TRANS, TRANSPOSE::CONJ_TRANS, ONE, singular_inv, u, ZERO, tmp);
  CUDABackend::matrix_mul(TRANSPOSE::NO_TRANS, TRANSPOSE::NO_TRANS, ONE, v, tmp, ZERO, result);
  cudaFree(d_s);
  cudaFree(info);
  cudaFree(workspace_buffer_on_device);
  free(workspace_buffer_on_host);
}
void CUDABackend::max_eigen_vector(const std::shared_ptr<MatrixXc> matrix, const std::shared_ptr<MatrixXc> ev) {
  const auto size = matrix->data.cols();

  double* d_w = nullptr;
  cudaMalloc((void**)&d_w, sizeof(double) * size);

  size_t workspace_in_bytes_on_device;
  size_t workspace_in_bytes_on_host;
  cusolverDnXsyevd_bufferSize(_handle_s, NULL, cusolverEigMode_t::CUSOLVER_EIG_MODE_VECTOR, cublasFillMode_t::CUBLAS_FILL_MODE_UPPER, size,
                              cudaDataType::CUDA_C_64F, matrix->ptr(), size, cudaDataType::CUDA_R_64F, d_w, cudaDataType::CUDA_C_64F,
                              &workspace_in_bytes_on_device, &workspace_in_bytes_on_host);

  void* workspace_buffer_on_device = nullptr;
  void* workspace_buffer_on_host = nullptr;
  cudaMalloc((void**)&workspace_buffer_on_device, workspace_in_bytes_on_device);
  if (workspace_in_bytes_on_host > 0) workspace_buffer_on_host = (void*)malloc(workspace_in_bytes_on_host);

  int* info;
  cudaMalloc((void**)&info, sizeof(int));
  cusolverDnXsyevd(_handle_s, NULL, cusolverEigMode_t::CUSOLVER_EIG_MODE_VECTOR, cublasFillMode_t::CUBLAS_FILL_MODE_UPPER, size,
                   cudaDataType::CUDA_C_64F, matrix->ptr(), size, cudaDataType::CUDA_R_64F, d_w, cudaDataType::CUDA_C_64F, workspace_buffer_on_device,
                   workspace_in_bytes_on_device, workspace_buffer_on_host, workspace_in_bytes_on_host, info);
  cudaFree(d_w);
  cudaFree(info);
  cudaFree(workspace_buffer_on_device);
  free(workspace_buffer_on_host);

  cudaMemcpy(ev->ptr(), matrix->ptr() + size * (size - 1), size * sizeof(complex), cudaMemcpyDeviceToDevice);
}

void CUDABackend::matrix_add(const double alpha, const std::shared_ptr<MatrixX> a, const std::shared_ptr<MatrixX> b) {
  cublasDaxpy_v2(_handle, static_cast<int>(a->data.size()), &alpha, a->ptr(), 1, b->ptr(), 1);
}
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
                             const std::shared_ptr<MatrixX> b, const double beta, const std::shared_ptr<MatrixX> c) {
  const auto lda = static_cast<int>(a->data.rows());
  const auto ldb = static_cast<int>(b->data.rows());
  const auto ldc = trans_a == TRANSPOSE::NO_TRANS ? static_cast<int>(a->data.rows()) : static_cast<int>(a->data.cols());
  const auto n = trans_b == TRANSPOSE::NO_TRANS ? static_cast<int>(b->data.cols()) : static_cast<int>(b->data.rows());
  const auto k = trans_a == TRANSPOSE::NO_TRANS ? static_cast<int>(a->data.cols()) : static_cast<int>(a->data.rows());
  cublasDgemm_v2(_handle, convert(trans_a), convert(trans_b), ldc, n, k, &alpha, a->ptr(), lda, b->ptr(), ldb, &beta, c->ptr(), ldc);
}

void CUDABackend::solve_g(const std::shared_ptr<MatrixX> a, const std::shared_ptr<MatrixX> b, const std::shared_ptr<MatrixX> c) {
  const auto n = static_cast<int>(a->data.cols());
  const auto lda = static_cast<int>(a->data.rows());
  const auto ldb = static_cast<int>(b->data.size());
  mat_cpy(b, c);

  size_t workspace_in_bytes_on_device;
  size_t workspace_in_bytes_on_host;
  cusolverDnXpotrf_bufferSize(_handle_s, NULL, cublasFillMode_t::CUBLAS_FILL_MODE_UPPER, n, cudaDataType::CUDA_R_64F, a->ptr(), lda,
                              cudaDataType::CUDA_R_64F, &workspace_in_bytes_on_device, &workspace_in_bytes_on_host);

  void* workspace_buffer_on_device = nullptr;
  void* workspace_buffer_on_host = nullptr;
  cudaMalloc((void**)&workspace_buffer_on_device, workspace_in_bytes_on_device);
  if (workspace_in_bytes_on_host > 0) workspace_buffer_on_host = (void*)malloc(workspace_in_bytes_on_host);

  int* info;
  cudaMalloc((void**)&info, sizeof(int));
  cusolverDnXpotrf(_handle_s, NULL, cublasFillMode_t::CUBLAS_FILL_MODE_UPPER, n, cudaDataType::CUDA_R_64F, a->ptr(), lda, cudaDataType::CUDA_R_64F,
                   workspace_buffer_on_device, workspace_in_bytes_on_device, workspace_buffer_on_host, workspace_in_bytes_on_host, info);

  cusolverDnXpotrs(_handle_s, NULL, cublasFillMode_t::CUBLAS_FILL_MODE_UPPER, n, 1, cudaDataType::CUDA_R_64F, a->ptr(), lda, cudaDataType::CUDA_R_64F,
                   c->ptr(), ldb, info);

  cudaFree(info);
  cudaFree(workspace_buffer_on_device);
  free(workspace_buffer_on_host);
}
void CUDABackend::solve_ch(const std::shared_ptr<MatrixXc> a, const std::shared_ptr<MatrixXc> b) {
  const auto n = static_cast<int>(a->data.cols());
  const auto lda = static_cast<int>(a->data.rows());
  const auto ldb = static_cast<int>(b->data.size());

  size_t workspace_in_bytes_on_device;
  size_t workspace_in_bytes_on_host;
  cusolverDnXpotrf_bufferSize(_handle_s, NULL, cublasFillMode_t::CUBLAS_FILL_MODE_UPPER, n, cudaDataType::CUDA_C_64F, a->ptr(), lda,
                              cudaDataType::CUDA_C_64F, &workspace_in_bytes_on_device, &workspace_in_bytes_on_host);

  void* workspace_buffer_on_device = nullptr;
  void* workspace_buffer_on_host = nullptr;
  cudaMalloc((void**)&workspace_buffer_on_device, workspace_in_bytes_on_device);
  if (workspace_in_bytes_on_host > 0) workspace_buffer_on_host = (void*)malloc(workspace_in_bytes_on_host);

  int* info;
  cudaMalloc((void**)&info, sizeof(int));
  cusolverDnXpotrf(_handle_s, NULL, cublasFillMode_t::CUBLAS_FILL_MODE_UPPER, n, cudaDataType::CUDA_C_64F, a->ptr(), lda, cudaDataType::CUDA_C_64F,
                   workspace_buffer_on_device, workspace_in_bytes_on_device, workspace_buffer_on_host, workspace_in_bytes_on_host, info);

  cusolverDnXpotrs(_handle_s, NULL, cublasFillMode_t::CUBLAS_FILL_MODE_UPPER, n, 1, cudaDataType::CUDA_C_64F, a->ptr(), lda, cudaDataType::CUDA_C_64F,
                   b->ptr(), ldb, info);

  cudaFree(info);
  cudaFree(workspace_buffer_on_device);
  free(workspace_buffer_on_host);
}
double CUDABackend::dot(const std::shared_ptr<MatrixX> a, const std::shared_ptr<MatrixX> b) {
  double d;
  cublasDdot_v2(_handle, static_cast<int>(a->data.size()), a->ptr(), 1, b->ptr(), 1, &d);
  return d;
}
complex CUDABackend::dot(const std::shared_ptr<MatrixXc> a, const std::shared_ptr<MatrixXc> b) {
  complex d;
  cublasZdotc_v2(_handle, static_cast<int>(a->data.size()), (const cuDoubleComplex*)a->ptr(), 1, (const cuDoubleComplex*)b->ptr(), 1,
                 (cuDoubleComplex*)&d);
  return d;
}
double CUDABackend::max_coefficient(const std::shared_ptr<MatrixXc> v) {
  int idx;
  cublasIzamax_v2(_handle, static_cast<int>(v->data.size()), (const cuDoubleComplex*)v->ptr(), 1, &idx);
  return std::abs(v->at(idx - 1, 0));  // 1-based indexing
}
double CUDABackend::max_coefficient(const std::shared_ptr<MatrixX> v) { return v->max_element(); }
std::shared_ptr<MatrixXc> CUDABackend::concat_row(const std::shared_ptr<MatrixXc> a, const std::shared_ptr<MatrixXc> b) {
  auto r = allocate_matrix_c("_cc_row", a->data.rows() + b->data.rows(), a->data.cols());
  for (int64_t i = 0; i < a->data.cols(); i++) {
    cudaMemcpy(r->ptr() + i * (a->data.rows() + b->data.rows()), a->ptr() + i * a->data.rows(), a->data.rows() * sizeof(complex),
               cudaMemcpyDeviceToDevice);
    cudaMemcpy(r->ptr() + i * (a->data.rows() + b->data.rows()) + a->data.rows(), b->ptr() + i * b->data.rows(), b->data.rows() * sizeof(complex),
               cudaMemcpyDeviceToDevice);
  }
  return r;
}
std::shared_ptr<MatrixXc> CUDABackend::concat_col(const std::shared_ptr<MatrixXc> a, const std::shared_ptr<MatrixXc> b) {
  auto r = allocate_matrix_c("_cc_col", a->data.rows(), a->data.cols() + b->data.cols());
  cudaMemcpy(r->ptr(), a->ptr(), a->data.rows() * a->data.cols() * sizeof(complex), cudaMemcpyDeviceToDevice);
  cudaMemcpy(r->ptr() + a->data.rows() * a->data.cols(), b->ptr(), b->data.rows() * b->data.cols() * sizeof(complex), cudaMemcpyDeviceToDevice);
  return r;
}
void CUDABackend::mat_cpy(const std::shared_ptr<MatrixX> a, const std::shared_ptr<MatrixX> b) {
  cublasDcopy_v2(_handle, static_cast<int>(a->data.rows()) * static_cast<int>(a->data.cols()), a->ptr(), 1, b->ptr(), 1);
}
void CUDABackend::mat_cpy(const std::shared_ptr<MatrixXc> a, const std::shared_ptr<MatrixXc> b) {
  cublasZcopy_v2(_handle, static_cast<int>(a->data.rows()) * static_cast<int>(a->data.cols()), (const cuDoubleComplex*)a->ptr(), 1,
                 (cuDoubleComplex*)b->ptr(), 1);
}

void CUDABackend::set_from_complex_drive(std::vector<core::DataArray>& data, const std::shared_ptr<MatrixXc> drive, const bool normalize,
                                         const double max_coefficient) {
  uint16_t* d_data = nullptr;
  cudaMalloc((void**)&d_data, data.size() * core::NUM_TRANS_IN_UNIT * sizeof(uint16_t));

  cu_set_from_complex_drive((const cuDoubleComplex*)drive->ptr(), (uint32_t)(data.size() * core::NUM_TRANS_IN_UNIT), normalize, max_coefficient,
                            d_data);
  for (size_t i = 0; i < data.size(); i++)
    cudaMemcpy(data[i].data(), d_data + i * core::NUM_TRANS_IN_UNIT, core::NUM_TRANS_IN_UNIT * sizeof(uint16_t), cudaMemcpyDeviceToHost);
}

std::shared_ptr<MatrixXc> CUDABackend::transfer_matrix(const double* foci, size_t foci_num, const std::vector<const double*>& positions,
                                                       const std::vector<const double*>& directions, double wavelength, double attenuation) {
  const auto m = static_cast<Eigen::Index>(foci_num);
  const auto n = static_cast<Eigen::Index>(positions.size() * core::NUM_TRANS_IN_UNIT);

  auto g = allocate_matrix_c("g", m, n);

  auto d_foci = allocate_matrix("_foci", 3, m);
  auto d_pos = allocate_matrix("_pos", 3, n);
  auto d_dir = allocate_matrix("_dir", 3, directions.size());
  cudaMemcpy(d_foci->ptr(), foci, m * 3 * sizeof(double), cudaMemcpyHostToDevice);
  for (size_t i = 0; i < positions.size(); i++)
    cudaMemcpy(d_pos->ptr() + core::NUM_TRANS_IN_UNIT * 3 * i, positions[i], core::NUM_TRANS_IN_UNIT * 3 * sizeof(double), cudaMemcpyHostToDevice);
  for (size_t i = 0; i < directions.size(); i++) cudaMemcpy(d_dir->ptr() + 3 * i, directions[i], 3 * sizeof(double), cudaMemcpyHostToDevice);

  cu_transfer_matrix((const double3*)d_foci->ptr(), (uint32_t)m, (const double3*)d_pos->ptr(), (const double3*)d_dir->ptr(), (uint32_t)n,
                     2.0 * M_PI / wavelength, attenuation, (cuDoubleComplex*)g->ptr());

  return g;
}

void CUDABackend::set_bcd_result(const std::shared_ptr<MatrixXc> mat, const std::shared_ptr<MatrixXc> vec, const size_t idx) {
  const uint32_t m = (uint32_t)vec->data.size();
  cu_set_bcd_result((const cuDoubleComplex*)vec->ptr(), m, (uint32_t)idx, (cuDoubleComplex*)mat->ptr());
}

std::shared_ptr<MatrixXc> CUDABackend::back_prop(const std::shared_ptr<MatrixXc> transfer, const std::shared_ptr<MatrixXc> amps) {
  const auto m = transfer->data.rows();
  const auto n = transfer->data.cols();

  auto denominator = allocate_matrix("denomi", m, 1);
  auto buffer = allocate_matrix("_bp_buf", m, 16);
  cu_col_sum_abs((const cuDoubleComplex*)transfer->ptr(), (uint32_t)m, (uint32_t)n, denominator->ptr(), buffer->ptr());

  auto b = allocate_matrix_c("b", n, m);
  cu_make_back_prop((const cuDoubleComplex*)amps->ptr(), denominator->ptr(), (const cuDoubleComplex*)transfer->ptr(), (uint32_t)m, (uint32_t)n,
                    (cuDoubleComplex*)b->ptr());
  return b;
}

std::shared_ptr<MatrixXc> CUDABackend::sigma_regularization(const std::shared_ptr<MatrixXc> transfer, const std::shared_ptr<MatrixXc> amps,
                                                            const double gamma) {
  const auto m = transfer->data.rows();
  const auto n = transfer->data.cols();

  auto tmp = allocate_matrix_c("_sr_tmp", n, 1);
  auto buffer = allocate_matrix("_sr_buffer", 16, n);
  cu_make_sigma_diagonal((const cuDoubleComplex*)transfer->ptr(), (uint32_t)m, (uint32_t)n, (const cuDoubleComplex*)amps->ptr(), gamma,
                         (cuDoubleComplex*)tmp->ptr(), buffer->ptr());

  auto sigma = allocate_matrix_c("sigma", n, n);
  sigma->fill(ZERO);
  sigma->set_diagonal(tmp);

  return sigma;
}

void CUDABackend::col_sum_imag(const std::shared_ptr<MatrixXc> mat, const std::shared_ptr<MatrixX> dst) {
  const auto m = mat->data.rows();
  const auto n = mat->data.cols();

  auto buffer = allocate_matrix("_csi_buf", m, 16);
  cu_col_sum_imag((const cuDoubleComplex*)mat->ptr(), (uint32_t)m, (uint32_t)n, dst->ptr(), buffer->ptr());
}

}  // namespace holo
}  // namespace gain
}  // namespace autd
