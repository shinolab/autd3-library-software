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
#include <thrust/extrema.h>
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

// CUDA currently does not support C++17
cublasHandle_t CuContext::handle = nullptr;
cusolverDnHandle_t CuContext::handle_s = nullptr;
size_t CuContext::cnt = 0;

void CuContext::init() {
  if (cnt++ > 0) return;
  cublasCreate_v2(&CuContext::handle);
  cusolverDnCreate(&CuContext::handle_s);
}

void CuContext::free() {
  if (--cnt > 0) return;
  cublasDestroy_v2(CuContext::handle);
  cusolverDnDestroy(CuContext::handle_s);
}

template <>
struct CuMatrix<double>::Impl {
  Impl(const size_t row, const size_t col) : _row(row), _col(col), _h_vec(nullptr), _d_vec(row * col) {}
  ~Impl() = default;

  const double* ptr() const { return _d_vec.data().get(); }
  double* ptr() { return _d_vec.data().get(); }

  void exp() { cu_exp((uint32_t)_row, (uint32_t)_col, ptr()); }
  void real(const std::shared_ptr<const CuMatrix<complex>> src) {
    cu_real((const cuDoubleComplex*)src->ptr(), (uint32_t)src->rows(), (uint32_t)src->cols(), ptr());
  }
  void scale(const double s) { cublasDscal_v2(CuContext::handle, static_cast<int>(_row * _col), &s, ptr(), 1); }
  void reciprocal(const std::shared_ptr<const CuMatrix<double>> src) {
    cu_reciprocal((uint32_t)src->rows(), (uint32_t)src->cols(), src->ptr(), ptr());
  }
  void abs(const std::shared_ptr<const CuMatrix<double>> src) { cu_abs((uint32_t)src->rows(), (uint32_t)src->cols(), src->ptr(), ptr()); }
  void hadamard_product(const std::shared_ptr<const CuMatrix<double>>& a, const std::shared_ptr<const CuMatrix<double>>& b) {
    cu_hadamard_product(a->ptr(), b->ptr(), (uint32_t)_row, (uint32_t)_col, ptr());
  }
  void pseudo_inverse_svd(const std::shared_ptr<CuMatrix<double>>& matrix, double alpha, const std::shared_ptr<CuMatrix<double>>& u,
                          const std::shared_ptr<CuMatrix<double>>& s, const std::shared_ptr<CuMatrix<double>>& v,
                          const std::shared_ptr<CuMatrix<double>>& buf) {
    const auto nc = matrix->cols();
    const auto nr = matrix->rows();

    const auto lda = static_cast<int>(nr);
    const auto ldu = static_cast<int>(nr);
    const auto ldv = static_cast<int>(nc);

    const auto s_size = std::min(nr, nc);
    double* d_s = nullptr;
    cudaMalloc((void**)&d_s, sizeof(double) * s_size);

    size_t workspace_in_bytes_on_device;
    size_t workspace_in_bytes_on_host;

    cusolverDnXgesvdp_bufferSize(CuContext::handle_s, NULL, cusolverEigMode_t::CUSOLVER_EIG_MODE_VECTOR, 0, static_cast<int>(nr),
                                 static_cast<int>(nc), cudaDataType::CUDA_R_64F, matrix->ptr(), lda, cudaDataType::CUDA_R_64F, d_s,
                                 cudaDataType::CUDA_R_64F, u->ptr(), ldu, cudaDataType::CUDA_R_64F, v->ptr(), ldv, cudaDataType::CUDA_R_64F,
                                 &workspace_in_bytes_on_device, &workspace_in_bytes_on_host);
    void* workspace_buffer_on_device = nullptr;
    void* workspace_buffer_on_host = nullptr;
    cudaMalloc((void**)&workspace_buffer_on_device, workspace_in_bytes_on_device);
    if (workspace_in_bytes_on_host > 0) workspace_buffer_on_host = (void*)malloc(workspace_in_bytes_on_host);

    int* info;
    cudaMalloc((void**)&info, sizeof(int));
    double h_err_sigma;
    cusolverDnXgesvdp(CuContext::handle_s, NULL, cusolverEigMode_t::CUSOLVER_EIG_MODE_VECTOR, 0, static_cast<int>(nr), static_cast<int>(nc),
                      cudaDataType::CUDA_R_64F, matrix->ptr(), lda, cudaDataType::CUDA_R_64F, d_s, cudaDataType::CUDA_R_64F, u->ptr(), ldu,
                      cudaDataType::CUDA_R_64F, v->ptr(), ldv, cudaDataType::CUDA_R_64F, workspace_buffer_on_device, workspace_in_bytes_on_device,
                      workspace_buffer_on_host, workspace_in_bytes_on_host, info, &h_err_sigma);

    calc_singular_inv(d_s, (uint32_t)s_size, alpha, s->ptr());

    buf->mul(TRANSPOSE::NO_TRANS, TRANSPOSE::TRANS, 1.0, s, u, 0.0);
    mul(TRANSPOSE::NO_TRANS, TRANSPOSE::NO_TRANS, 1.0, v, buf, 0.0);
    cudaFree(d_s);
    cudaFree(info);
    cudaFree(workspace_buffer_on_device);
    free(workspace_buffer_on_host);
  }

  void add(const double alpha, const std::shared_ptr<CuMatrix<double>>& a) {
    cublasDaxpy_v2(CuContext::handle, static_cast<int>(a->rows() * a->cols()), &alpha, a->ptr(), 1, ptr(), 1);
  }
  void mul(const TRANSPOSE trans_a, const TRANSPOSE trans_b, const double alpha, const std::shared_ptr<const CuMatrix<double>>& a,
           const std::shared_ptr<const CuMatrix<double>>& b, const double beta) {
    const auto lda = static_cast<int>(a->rows());
    const auto ldb = static_cast<int>(b->rows());
    const auto ldc = trans_a == TRANSPOSE::NO_TRANS ? static_cast<int>(a->rows()) : static_cast<int>(a->cols());
    const auto n = trans_b == TRANSPOSE::NO_TRANS ? static_cast<int>(b->cols()) : static_cast<int>(b->rows());
    const auto k = trans_a == TRANSPOSE::NO_TRANS ? static_cast<int>(a->cols()) : static_cast<int>(a->rows());
    cublasDgemm_v2(CuContext::handle, convert(trans_a), convert(trans_b), ldc, n, k, &alpha, a->ptr(), lda, b->ptr(), ldb, &beta, ptr(), ldc);
  }
  void solve(const std::shared_ptr<CuMatrix<double>>& b) {
    const auto n = static_cast<int>(_col);
    const auto lda = static_cast<int>(_row);
    const auto ldb = static_cast<int>(b->rows());

    size_t workspace_in_bytes_on_device;
    size_t workspace_in_bytes_on_host;
    cusolverDnXpotrf_bufferSize(CuContext::handle_s, NULL, cublasFillMode_t::CUBLAS_FILL_MODE_UPPER, n, cudaDataType::CUDA_R_64F, ptr(), lda,
                                cudaDataType::CUDA_R_64F, &workspace_in_bytes_on_device, &workspace_in_bytes_on_host);

    void* workspace_buffer_on_device = nullptr;
    void* workspace_buffer_on_host = nullptr;
    cudaMalloc((void**)&workspace_buffer_on_device, workspace_in_bytes_on_device);
    if (workspace_in_bytes_on_host > 0) workspace_buffer_on_host = (void*)malloc(workspace_in_bytes_on_host);

    int* info;
    cudaMalloc((void**)&info, sizeof(int));
    cusolverDnXpotrf(CuContext::handle_s, NULL, cublasFillMode_t::CUBLAS_FILL_MODE_UPPER, n, cudaDataType::CUDA_R_64F, ptr(), lda,
                     cudaDataType::CUDA_R_64F, workspace_buffer_on_device, workspace_in_bytes_on_device, workspace_buffer_on_host,
                     workspace_in_bytes_on_host, info);

    cusolverDnXpotrs(CuContext::handle_s, NULL, cublasFillMode_t::CUBLAS_FILL_MODE_UPPER, n, 1, cudaDataType::CUDA_R_64F, ptr(), lda,
                     cudaDataType::CUDA_R_64F, b->ptr(), ldb, info);

    cudaFree(info);
    cudaFree(workspace_buffer_on_device);
    free(workspace_buffer_on_host);
  }

  double dot(const std::shared_ptr<const CuMatrix<double>> b) {
    double d;
    cublasDdot_v2(CuContext::handle, static_cast<int>(_row * _col), ptr(), 1, b->ptr(), 1, &d);
    return d;
  }
  double max_element() const { return *thrust::max_element(_d_vec.begin(), _d_vec.end()); }
  void concat_row(const std::shared_ptr<const CuMatrix<double>>& a, const std::shared_ptr<const CuMatrix<double>>& b) {
    for (int64_t i = 0; i < a->cols(); i++) {
      cudaMemcpy(ptr() + i * (a->rows() + b->rows()), a->ptr() + i * a->rows(), a->rows() * sizeof(double), cudaMemcpyDeviceToDevice);
      cudaMemcpy(ptr() + i * (a->rows() + b->rows()) + a->rows(), b->ptr() + i * b->rows(), b->rows() * sizeof(double), cudaMemcpyDeviceToDevice);
    }
  }
  void concat_col(const std::shared_ptr<const CuMatrix<double>>& a, const std::shared_ptr<const CuMatrix<double>>& b) {
    cudaMemcpy(ptr(), a->ptr(), a->rows() * a->cols() * sizeof(double), cudaMemcpyDeviceToDevice);
    cudaMemcpy(ptr() + a->rows() * a->cols(), b->ptr(), b->rows() * b->cols() * sizeof(double), cudaMemcpyDeviceToDevice);
  }

  double at(const size_t row, const size_t col) {
    double v;
    cudaMemcpy(&v, _d_vec.data().get() + row + _row * col, sizeof(double), cudaMemcpyDeviceToHost);
    return v;
  }
  void set(const size_t row, const size_t col, double v) {
    cudaMemcpy(_d_vec.data().get() + row + _row * col, &v, sizeof(double), cudaMemcpyHostToDevice);
  }
  void get_col(const std::shared_ptr<const CuMatrix<double>>& src, const size_t i) {
    cudaMemcpy(ptr(), src->ptr() + i * _row, _row * sizeof(double), cudaMemcpyDeviceToDevice);
  }
  void fill(double v) { thrust::fill(_d_vec.begin(), _d_vec.end(), v); }
  void get_diagonal(const std::shared_ptr<const CuMatrix<double>>& src) {
    cu_get_diagonal(src->ptr(), ptr(), (uint32_t)(std::min)(src->rows(), src->cols()));
  }
  void create_diagonal(const std::shared_ptr<const CuMatrix<double>>& v) {
    fill(0.0);
    cu_set_diagonal(v->ptr(), ptr(), (uint32_t)(std::min)(_row, _col));
  }
  void copy_from(const std::shared_ptr<const CuMatrix<double>>& src) {
    cudaMemcpy(_d_vec.data().get(), src->ptr(), _row * _col * sizeof(double), cudaMemcpyDeviceToDevice);
  }
  void copy_from(const double* v, const size_t n) { cudaMemcpy(_d_vec.data().get(), v, n * sizeof(double), cudaMemcpyHostToDevice); }
  void copy_to_host() {
    if (_h_vec == nullptr) _h_vec = std::make_unique<double[]>(_row * _col);
    cudaMemcpy(_h_vec.get(), _d_vec.data().get(), _row * _col * sizeof(double), cudaMemcpyDeviceToHost);
  }

  void set_from_arg(std::vector<core::DataArray>& data, const size_t n) {
    uint16_t* d_data = nullptr;
    cudaMalloc((void**)&d_data, data.size() * core::NUM_TRANS_IN_UNIT * sizeof(uint16_t));
    cudaMemset(d_data, 0, data.size() * core::NUM_TRANS_IN_UNIT * sizeof(uint16_t));

    cu_set_from_arg(ptr(), (uint32_t)n, d_data);
    for (size_t i = 0; i < data.size(); i++)
      cudaMemcpy(data[i].data(), d_data + i * core::NUM_TRANS_IN_UNIT, core::NUM_TRANS_IN_UNIT * sizeof(uint16_t), cudaMemcpyDeviceToHost);
  }
  void col_sum_imag(const std::shared_ptr<const CuMatrix<complex>> mat) {
    const auto m = mat->rows();
    const auto n = mat->cols();
    thrust::device_vector<double> buffer(m * BLOCK_SIZE / 2);
    cu_col_sum_imag((const cuDoubleComplex*)mat->ptr(), (uint32_t)m, (uint32_t)n, ptr(), buffer.data().get());
  }

 private:
  size_t _row;
  size_t _col;
  std::unique_ptr<double[]> _h_vec;
  thrust::device_vector<double> _d_vec;
};

template <>
struct CuMatrix<complex>::Impl {
  Impl(const size_t row, const size_t col) : _row(row), _col(col), _h_vec(nullptr), _d_vec(row * col) {}
  ~Impl() = default;

  const complex* ptr() const { return _d_vec.data().get(); }
  complex* ptr() { return _d_vec.data().get(); }

  void make_complex(const std::shared_ptr<const CuMatrix<double>>& r, const std::shared_ptr<const CuMatrix<double>>& i)

  {
    cu_make_complex(r->ptr(), i->ptr(), (uint32_t)_row, (uint32_t)_col, (cuDoubleComplex*)ptr());
  }
  void exp() { cu_exp((uint32_t)_row, (uint32_t)_col, (cuDoubleComplex*)ptr()); }
  void scale(const complex s) {
    cublasZscal_v2(CuContext::handle, static_cast<int>(_row * _col), (const cuDoubleComplex*)&s, (cuDoubleComplex*)ptr(), 1);
  }
  void reciprocal(const std::shared_ptr<const CuMatrix<complex>> src) {
    cu_reciprocal((uint32_t)src->rows(), (uint32_t)src->cols(), (const cuDoubleComplex*)src->ptr(), (cuDoubleComplex*)ptr());
  }
  void abs(const std::shared_ptr<const CuMatrix<complex>> src) {
    cu_abs((uint32_t)src->rows(), (uint32_t)src->cols(), (const cuDoubleComplex*)src->ptr(), (cuDoubleComplex*)ptr());
  }
  void arg(const std::shared_ptr<const CuMatrix<complex>> src) {
    cu_arg((const cuDoubleComplex*)src->ptr(), (uint32_t)src->rows(), (uint32_t)src->cols(), (cuDoubleComplex*)ptr());
  }
  void hadamard_product(const std::shared_ptr<const CuMatrix<complex>>& a, const std::shared_ptr<const CuMatrix<complex>>& b) {
    cu_hadamard_product((const cuDoubleComplex*)a->ptr(), (const cuDoubleComplex*)b->ptr(), (uint32_t)_row, (uint32_t)_col, (cuDoubleComplex*)ptr());
  }
  void pseudo_inverse_svd(const std::shared_ptr<CuMatrix<complex>>& matrix, double alpha, const std::shared_ptr<CuMatrix<complex>>& u,
                          const std::shared_ptr<CuMatrix<complex>>& s, const std::shared_ptr<CuMatrix<complex>>& v,
                          const std::shared_ptr<CuMatrix<complex>>& buf) {
    const auto nc = matrix->cols();
    const auto nr = matrix->rows();

    const auto lda = static_cast<int>(nr);
    const auto ldu = static_cast<int>(nr);
    const auto ldv = static_cast<int>(nc);

    const auto s_size = std::min(nr, nc);
    double* d_s = nullptr;
    cudaMalloc((void**)&d_s, sizeof(double) * s_size);

    size_t workspace_in_bytes_on_device;
    size_t workspace_in_bytes_on_host;

    cusolverDnXgesvdp_bufferSize(CuContext::handle_s, NULL, cusolverEigMode_t::CUSOLVER_EIG_MODE_VECTOR, 0, static_cast<int>(nr),
                                 static_cast<int>(nc), cudaDataType::CUDA_C_64F, matrix->ptr(), lda, cudaDataType::CUDA_R_64F, d_s,
                                 cudaDataType::CUDA_C_64F, u->ptr(), ldu, cudaDataType::CUDA_C_64F, v->ptr(), ldv, cudaDataType::CUDA_C_64F,
                                 &workspace_in_bytes_on_device, &workspace_in_bytes_on_host);
    void* workspace_buffer_on_device = nullptr;
    void* workspace_buffer_on_host = nullptr;
    cudaMalloc((void**)&workspace_buffer_on_device, workspace_in_bytes_on_device);
    if (workspace_in_bytes_on_host > 0) workspace_buffer_on_host = (void*)malloc(workspace_in_bytes_on_host);

    int* info;
    cudaMalloc((void**)&info, sizeof(int));
    double h_err_sigma;
    cusolverDnXgesvdp(CuContext::handle_s, NULL, cusolverEigMode_t::CUSOLVER_EIG_MODE_VECTOR, 0, static_cast<int>(nr), static_cast<int>(nc),
                      cudaDataType::CUDA_C_64F, matrix->ptr(), lda, cudaDataType::CUDA_R_64F, d_s, cudaDataType::CUDA_C_64F, u->ptr(), ldu,
                      cudaDataType::CUDA_C_64F, v->ptr(), ldv, cudaDataType::CUDA_C_64F, workspace_buffer_on_device, workspace_in_bytes_on_device,
                      workspace_buffer_on_host, workspace_in_bytes_on_host, info, &h_err_sigma);

    calc_singular_inv(d_s, (uint32_t)s_size, alpha, (cuDoubleComplex*)s->ptr());

    buf->mul(TRANSPOSE::NO_TRANS, TRANSPOSE::CONJ_TRANS, ONE, s, u, ZERO);
    mul(TRANSPOSE::NO_TRANS, TRANSPOSE::NO_TRANS, ONE, v, buf, ZERO);
    cudaFree(d_s);
    cudaFree(info);
    cudaFree(workspace_buffer_on_device);
    free(workspace_buffer_on_host);
  }
  void max_eigen_vector(const std::shared_ptr<CuMatrix<complex>>& ev) {
    const auto size = _col;
    double* d_w = nullptr;
    cudaMalloc((void**)&d_w, sizeof(double) * size);

    size_t workspace_in_bytes_on_device;
    size_t workspace_in_bytes_on_host;
    cusolverDnXsyevd_bufferSize(CuContext::handle_s, NULL, cusolverEigMode_t::CUSOLVER_EIG_MODE_VECTOR, cublasFillMode_t::CUBLAS_FILL_MODE_UPPER,
                                size, cudaDataType::CUDA_C_64F, ptr(), size, cudaDataType::CUDA_R_64F, d_w, cudaDataType::CUDA_C_64F,
                                &workspace_in_bytes_on_device, &workspace_in_bytes_on_host);

    void* workspace_buffer_on_device = nullptr;
    void* workspace_buffer_on_host = nullptr;
    cudaMalloc((void**)&workspace_buffer_on_device, workspace_in_bytes_on_device);
    if (workspace_in_bytes_on_host > 0) workspace_buffer_on_host = (void*)malloc(workspace_in_bytes_on_host);

    int* info;
    cudaMalloc((void**)&info, sizeof(int));
    cusolverDnXsyevd(CuContext::handle_s, NULL, cusolverEigMode_t::CUSOLVER_EIG_MODE_VECTOR, cublasFillMode_t::CUBLAS_FILL_MODE_UPPER, size,
                     cudaDataType::CUDA_C_64F, ptr(), size, cudaDataType::CUDA_R_64F, d_w, cudaDataType::CUDA_C_64F, workspace_buffer_on_device,
                     workspace_in_bytes_on_device, workspace_buffer_on_host, workspace_in_bytes_on_host, info);
    cudaFree(d_w);
    cudaFree(info);
    cudaFree(workspace_buffer_on_device);
    free(workspace_buffer_on_host);

    cudaMemcpy(ev->ptr(), ptr() + size * (size - 1), size * sizeof(complex), cudaMemcpyDeviceToDevice);
  }
  void add(const complex alpha, const std::shared_ptr<CuMatrix<complex>>& a) {
    cublasZaxpy_v2(CuContext::handle, static_cast<int>(a->rows() * a->cols()), (const cuDoubleComplex*)&alpha, (const cuDoubleComplex*)a->ptr(), 1,
                   (cuDoubleComplex*)ptr(), 1);
  }
  void mul(const TRANSPOSE trans_a, const TRANSPOSE trans_b, const complex alpha, const std::shared_ptr<const CuMatrix<complex>>& a,
           const std::shared_ptr<const CuMatrix<complex>>& b, const complex beta) {
    const auto lda = static_cast<int>(a->rows());
    const auto ldb = static_cast<int>(b->rows());
    const auto ldc = trans_a == TRANSPOSE::NO_TRANS ? static_cast<int>(a->rows()) : static_cast<int>(a->cols());
    const auto n = trans_b == TRANSPOSE::NO_TRANS ? static_cast<int>(b->cols()) : static_cast<int>(b->rows());
    const auto k = trans_a == TRANSPOSE::NO_TRANS ? static_cast<int>(a->cols()) : static_cast<int>(a->rows());
    cublasZgemm_v2(CuContext::handle, convert(trans_a), convert(trans_b), ldc, n, k, (const cuDoubleComplex*)&alpha, (const cuDoubleComplex*)a->ptr(),
                   lda, (const cuDoubleComplex*)b->ptr(), ldb, (const cuDoubleComplex*)&beta, (cuDoubleComplex*)ptr(), ldc);
  }
  void solve(const std::shared_ptr<CuMatrix<complex>>& b) {
    const auto n = static_cast<int>(_col);
    const auto lda = static_cast<int>(_row);
    const auto ldb = static_cast<int>(b->rows());

    size_t workspace_in_bytes_on_device;
    size_t workspace_in_bytes_on_host;
    cusolverDnXpotrf_bufferSize(CuContext::handle_s, NULL, cublasFillMode_t::CUBLAS_FILL_MODE_UPPER, n, cudaDataType::CUDA_C_64F, ptr(), lda,
                                cudaDataType::CUDA_C_64F, &workspace_in_bytes_on_device, &workspace_in_bytes_on_host);

    void* workspace_buffer_on_device = nullptr;
    void* workspace_buffer_on_host = nullptr;
    cudaMalloc((void**)&workspace_buffer_on_device, workspace_in_bytes_on_device);
    if (workspace_in_bytes_on_host > 0) workspace_buffer_on_host = (void*)malloc(workspace_in_bytes_on_host);

    int* info;
    cudaMalloc((void**)&info, sizeof(int));
    cusolverDnXpotrf(CuContext::handle_s, NULL, cublasFillMode_t::CUBLAS_FILL_MODE_UPPER, n, cudaDataType::CUDA_C_64F, ptr(), lda,
                     cudaDataType::CUDA_C_64F, workspace_buffer_on_device, workspace_in_bytes_on_device, workspace_buffer_on_host,
                     workspace_in_bytes_on_host, info);
    cusolverDnXpotrs(CuContext::handle_s, NULL, cublasFillMode_t::CUBLAS_FILL_MODE_UPPER, n, 1, cudaDataType::CUDA_C_64F, ptr(), lda,
                     cudaDataType::CUDA_C_64F, b->ptr(), ldb, info);

    cudaFree(info);
    cudaFree(workspace_buffer_on_device);
    free(workspace_buffer_on_host);
  }
  complex dot(const std::shared_ptr<const CuMatrix<complex>> b) {
    complex d;
    cublasZdotc_v2(CuContext::handle, static_cast<int>(_row * _col), (const cuDoubleComplex*)ptr(), 1, (const cuDoubleComplex*)b->ptr(), 1,
                   (cuDoubleComplex*)&d);
    return d;
  }
  double max_element() const {
    int idx;
    cublasIzamax_v2(CuContext::handle, static_cast<int>(_row * _col), (const cuDoubleComplex*)ptr(), 1, &idx);
    complex c;
    cudaMemcpy(&c, _d_vec.data().get() + idx - 1, sizeof(complex), cudaMemcpyDeviceToHost);  // 1-based indexing
    return std::abs(c);
  }
  void concat_row(const std::shared_ptr<const CuMatrix<complex>>& a, const std::shared_ptr<const CuMatrix<complex>>& b) {
    for (int64_t i = 0; i < a->cols(); i++) {
      cudaMemcpy(ptr() + i * (a->rows() + b->rows()), a->ptr() + i * a->rows(), a->rows() * sizeof(complex), cudaMemcpyDeviceToDevice);
      cudaMemcpy(ptr() + i * (a->rows() + b->rows()) + a->rows(), b->ptr() + i * b->rows(), b->rows() * sizeof(complex), cudaMemcpyDeviceToDevice);
    }
  }
  void concat_col(const std::shared_ptr<const CuMatrix<complex>>& a, const std::shared_ptr<const CuMatrix<complex>>& b) {
    cudaMemcpy(ptr(), a->ptr(), a->rows() * a->cols() * sizeof(complex), cudaMemcpyDeviceToDevice);
    cudaMemcpy(ptr() + a->rows() * a->cols(), b->ptr(), b->rows() * b->cols() * sizeof(complex), cudaMemcpyDeviceToDevice);
  }

  complex at(const size_t row, const size_t col) {
    complex v;
    cudaMemcpy(&v, _d_vec.data().get() + row + _row * col, sizeof(complex), cudaMemcpyDeviceToHost);
    return v;
  }
  void set(const size_t row, const size_t col, complex v) {
    cudaMemcpy(_d_vec.data().get() + row + _row * col, &v, sizeof(complex), cudaMemcpyHostToDevice);
  }
  void get_col(const std::shared_ptr<const CuMatrix<complex>>& src, const size_t i) {
    cudaMemcpy(ptr(), src->ptr() + i * _row, _row * sizeof(complex), cudaMemcpyDeviceToDevice);
  }
  void fill(complex v) { thrust::fill(_d_vec.begin(), _d_vec.end(), v); }
  void get_diagonal(const std::shared_ptr<const CuMatrix<complex>>& src) {
    cu_get_diagonal(src->ptr(), ptr(), (uint32_t)(std::min)(src->rows(), src->cols()));
  }
  void create_diagonal(const std::shared_ptr<const CuMatrix<complex>>& v) {
    fill(ZERO);
    cu_set_diagonal(v->ptr(), ptr(), (uint32_t)(std::min)(_row, _col));
  }
  void create_diagonal(const thrust::device_vector<complex>& v) {
    fill(ZERO);
    cu_set_diagonal(v.data().get(), ptr(), (uint32_t)(std::min)(_row, _col));
  }
  void copy_from(const std::shared_ptr<const CuMatrix<complex>>& src) {
    cudaMemcpy(_d_vec.data().get(), src->ptr(), _row * _col * sizeof(complex), cudaMemcpyDeviceToDevice);
  }
  void copy_from(const complex* v, const size_t n) { cudaMemcpy(_d_vec.data().get(), v, n * sizeof(complex), cudaMemcpyHostToDevice); }
  void copy_to_host() {
    if (_h_vec == nullptr) _h_vec = std::make_unique<complex[]>(_row * _col);
    cudaMemcpy(_h_vec.get(), _d_vec.data().get(), _row * _col * sizeof(complex), cudaMemcpyDeviceToHost);
  }

  void set_from_complex_drive(std::vector<core::DataArray>& data, const bool normalize, const double max_coefficient) {
    uint16_t* d_data = nullptr;
    cudaMalloc((void**)&d_data, data.size() * core::NUM_TRANS_IN_UNIT * sizeof(uint16_t));

    cu_set_from_complex_drive((const cuDoubleComplex*)ptr(), (uint32_t)(data.size() * core::NUM_TRANS_IN_UNIT), normalize, max_coefficient, d_data);
    for (size_t i = 0; i < data.size(); i++)
      cudaMemcpy(data[i].data(), d_data + i * core::NUM_TRANS_IN_UNIT, core::NUM_TRANS_IN_UNIT * sizeof(uint16_t), cudaMemcpyDeviceToHost);
  }

  void transfer_matrix(const double* foci, size_t foci_num, const std::vector<const double*>& positions, const std::vector<const double*>& directions,
                       double wavelength, double attenuation) {
    const auto m = foci_num;
    const auto n = positions.size() * core::NUM_TRANS_IN_UNIT;

    thrust::device_vector<double3> d_foci(m);
    thrust::device_vector<double3> d_pos(n);
    thrust::device_vector<double3> d_dir(directions.size());
    cudaMemcpy(d_foci.data().get(), foci, m * sizeof(double3), cudaMemcpyHostToDevice);
    for (size_t i = 0; i < positions.size(); i++)
      cudaMemcpy(d_pos.data().get() + core::NUM_TRANS_IN_UNIT * i, positions[i], core::NUM_TRANS_IN_UNIT * sizeof(double3), cudaMemcpyHostToDevice);
    for (size_t i = 0; i < directions.size(); i++) cudaMemcpy(d_dir.data().get() + i, directions[i], sizeof(double3), cudaMemcpyHostToDevice);

    cu_transfer_matrix(d_foci.data().get(), (uint32_t)m, d_pos.data().get(), d_dir.data().get(), (uint32_t)n, 2.0 * M_PI / wavelength, attenuation,
                       (cuDoubleComplex*)ptr());
  }
  void set_bcd_result(std::shared_ptr<const CuMatrix<complex>> vec, const size_t index) {
    const uint32_t m = (uint32_t)vec->rows();
    cu_set_bcd_result((const cuDoubleComplex*)vec->ptr(), m, (uint32_t)index, (cuDoubleComplex*)ptr());
  }
  void back_prop(const std::shared_ptr<const CuMatrix<complex>>& transfer, const std::shared_ptr<const CuMatrix<complex>>& amps) {
    const auto m = transfer->rows();
    const auto n = transfer->cols();

    thrust::device_vector<double> denominator(m);
    thrust::device_vector<double> buffer(m * BLOCK_SIZE / 2);
    for (int i = 0; i < m; i++) {
      double v;
      cudaMemcpy(&v, denominator.data().get() + i, sizeof(double), cudaMemcpyDeviceToHost);
    }
    cu_col_sum_abs((const cuDoubleComplex*)transfer->ptr(), (uint32_t)m, (uint32_t)n, denominator.data().get(), buffer.data().get());
    for (int i = 0; i < m; i++) {
      double v;
      cudaMemcpy(&v, denominator.data().get() + i, sizeof(double), cudaMemcpyDeviceToHost);
    }
    cu_make_back_prop((const cuDoubleComplex*)amps->ptr(), denominator.data().get(), (const cuDoubleComplex*)transfer->ptr(), (uint32_t)m,
                      (uint32_t)n, (cuDoubleComplex*)ptr());
  }

  void sigma_regularization(const std::shared_ptr<const CuMatrix<complex>>& transfer, const std::shared_ptr<const CuMatrix<complex>>& amps,
                            const double gamma) {
    const auto m = transfer->rows();
    const auto n = transfer->cols();

    thrust::device_vector<complex> tmp(n);
    thrust::device_vector<double> buffer(n * BLOCK_SIZE / 2);
    cu_make_sigma_diagonal((const cuDoubleComplex*)transfer->ptr(), (uint32_t)m, (uint32_t)n, (const cuDoubleComplex*)amps->ptr(), gamma,
                           (cuDoubleComplex*)tmp.data().get(), buffer.data().get());
    create_diagonal(tmp);
  }

 private:
  size_t _row;
  size_t _col;
  std::unique_ptr<complex[]> _h_vec;
  thrust::device_vector<complex> _d_vec;
};

CuMatrix<double>::CuMatrix(const size_t row, const size_t col) : _row(row), _col(col), _pimpl(std::make_unique<CuMatrix<double>::Impl>(row, col)) {}
CuMatrix<complex>::CuMatrix(const size_t row, const size_t col) : _row(row), _col(col), _pimpl(std::make_unique<CuMatrix<complex>::Impl>(row, col)) {}

template <>
const double* CuMatrix<double>::ptr() const {
  return _pimpl->ptr();
}
template <>
const complex* CuMatrix<complex>::ptr() const {
  return _pimpl->ptr();
}
template <>
double* CuMatrix<double>::ptr() {
  return _pimpl->ptr();
}
template <>
complex* CuMatrix<complex>::ptr() {
  return _pimpl->ptr();
}

void CuMatrix<double>::make_complex(const std::shared_ptr<const CuMatrix<double>>& r, const std::shared_ptr<const CuMatrix<double>>& i) {}
void CuMatrix<complex>::make_complex(const std::shared_ptr<const CuMatrix<double>>& r, const std::shared_ptr<const CuMatrix<double>>& i) {
  _pimpl->make_complex(r, i);
}
void CuMatrix<double>::exp() { _pimpl->exp(); }
void CuMatrix<complex>::exp() { _pimpl->exp(); }
void CuMatrix<double>::scale(const double s) { _pimpl->scale(s); }
void CuMatrix<complex>::scale(const complex s) { _pimpl->scale(s); }
void CuMatrix<double>::reciprocal(const std::shared_ptr<const CuMatrix<double>>& src) { _pimpl->reciprocal(src); }
void CuMatrix<complex>::reciprocal(const std::shared_ptr<const CuMatrix<complex>>& src) { _pimpl->reciprocal(src); }
void CuMatrix<double>::abs(const std::shared_ptr<const CuMatrix<double>>& src) { _pimpl->abs(src); }
void CuMatrix<complex>::abs(const std::shared_ptr<const CuMatrix<complex>>& src) { _pimpl->abs(src); }
void CuMatrix<double>::real(const std::shared_ptr<const CuMatrix<complex>>& src) { _pimpl->real(src); }
void CuMatrix<complex>::real(const std::shared_ptr<const CuMatrix<complex>>& src) {}
void CuMatrix<double>::arg(const std::shared_ptr<const CuMatrix<complex>>& src) {}
void CuMatrix<complex>::arg(const std::shared_ptr<const CuMatrix<complex>>& src) { _pimpl->arg(src); }
void CuMatrix<double>::hadamard_product(const std::shared_ptr<const CuMatrix<double>>& a, const std::shared_ptr<const CuMatrix<double>>& b) {
  _pimpl->hadamard_product(a, b);
}
void CuMatrix<complex>::hadamard_product(const std::shared_ptr<const CuMatrix<complex>>& a, const std::shared_ptr<const CuMatrix<complex>>& b) {
  _pimpl->hadamard_product(a, b);
}
void CuMatrix<double>::pseudo_inverse_svd(const std::shared_ptr<CuMatrix<double>>& matrix, double alpha, const std::shared_ptr<CuMatrix<double>>& u,
                                          const std::shared_ptr<CuMatrix<double>>& s, const std::shared_ptr<CuMatrix<double>>& vt,
                                          const std::shared_ptr<CuMatrix<double>>& buf) {
  _pimpl->pseudo_inverse_svd(matrix, alpha, u, s, vt, buf);
}
void CuMatrix<complex>::pseudo_inverse_svd(const std::shared_ptr<CuMatrix<complex>>& matrix, double alpha,
                                           const std::shared_ptr<CuMatrix<complex>>& u, const std::shared_ptr<CuMatrix<complex>>& s,
                                           const std::shared_ptr<CuMatrix<complex>>& vt, const std::shared_ptr<CuMatrix<complex>>& buf) {
  _pimpl->pseudo_inverse_svd(matrix, alpha, u, s, vt, buf);
}
void CuMatrix<double>::max_eigen_vector(const std::shared_ptr<CuMatrix<double>>& ev) {}
void CuMatrix<complex>::max_eigen_vector(const std::shared_ptr<CuMatrix<complex>>& ev) { _pimpl->max_eigen_vector(ev); }
void CuMatrix<double>::add(const double alpha, const std::shared_ptr<CuMatrix<double>>& a) { _pimpl->add(alpha, a); }
void CuMatrix<complex>::add(const complex alpha, const std::shared_ptr<CuMatrix<complex>>& a) { _pimpl->add(alpha, a); }
template <>
void CuMatrix<double>::mul(const TRANSPOSE trans_a, const TRANSPOSE trans_b, const double alpha, const std::shared_ptr<const CuMatrix<double>>& a,
                           const std::shared_ptr<const CuMatrix<double>>& b, const double beta) {
  _pimpl->mul(trans_a, trans_b, alpha, a, b, beta);
}
template <>
void CuMatrix<complex>::mul(const TRANSPOSE trans_a, const TRANSPOSE trans_b, const complex alpha, const std::shared_ptr<const CuMatrix<complex>>& a,
                            const std::shared_ptr<const CuMatrix<complex>>& b, const complex beta) {
  _pimpl->mul(trans_a, trans_b, alpha, a, b, beta);
}
void CuMatrix<double>::solve(const std::shared_ptr<CuMatrix<double>>& b) { _pimpl->solve(b); }
void CuMatrix<complex>::solve(const std::shared_ptr<CuMatrix<complex>>& b) { _pimpl->solve(b); }
double CuMatrix<double>::dot(const std::shared_ptr<const CuMatrix<double>>& b) { return _pimpl->dot(b); }
complex CuMatrix<complex>::dot(const std::shared_ptr<const CuMatrix<complex>>& b) { return _pimpl->dot(b); }
double CuMatrix<double>::max_element() const { return _pimpl->max_element(); }
double CuMatrix<complex>::max_element() const { return _pimpl->max_element(); }
void CuMatrix<double>::concat_col(const std::shared_ptr<const CuMatrix<double>>& a, const std::shared_ptr<const CuMatrix<double>>& b) {
  _pimpl->concat_col(a, b);
}
void CuMatrix<complex>::concat_col(const std::shared_ptr<const CuMatrix<complex>>& a, const std::shared_ptr<const CuMatrix<complex>>& b) {
  _pimpl->concat_col(a, b);
}
void CuMatrix<double>::concat_row(const std::shared_ptr<const CuMatrix<double>>& a, const std::shared_ptr<const CuMatrix<double>>& b) {
  _pimpl->concat_row(a, b);
}
void CuMatrix<complex>::concat_row(const std::shared_ptr<const CuMatrix<complex>>& a, const std::shared_ptr<const CuMatrix<complex>>& b) {
  _pimpl->concat_row(a, b);
}

double CuMatrix<double>::at(const size_t row, const size_t col) { return _pimpl->at(row, col); }
complex CuMatrix<complex>::at(const size_t row, const size_t col) { return _pimpl->at(row, col); }
void CuMatrix<double>::set(const size_t row, const size_t col, double v) { _pimpl->set(row, col, v); }
void CuMatrix<complex>::set(const size_t row, const size_t col, complex v) { _pimpl->set(row, col, v); }
void CuMatrix<double>::get_col(const std::shared_ptr<const CuMatrix<double>>& src, const size_t i) { _pimpl->get_col(src, i); }
void CuMatrix<complex>::get_col(const std::shared_ptr<const CuMatrix<complex>>& src, const size_t i) { _pimpl->get_col(src, i); }
void CuMatrix<double>::fill(double v) { _pimpl->fill(v); }
void CuMatrix<complex>::fill(complex v) { _pimpl->fill(v); }
void CuMatrix<double>::get_diagonal(const std::shared_ptr<const CuMatrix<double>>& src) { return _pimpl->get_diagonal(src); }
void CuMatrix<complex>::get_diagonal(const std::shared_ptr<const CuMatrix<complex>>& src) { return _pimpl->get_diagonal(src); }
void CuMatrix<double>::create_diagonal(const std::shared_ptr<const CuMatrix<double>>& v) { _pimpl->create_diagonal(v); }
void CuMatrix<complex>::create_diagonal(const std::shared_ptr<const CuMatrix<complex>>& v) { _pimpl->create_diagonal(v); }
void CuMatrix<double>::copy_from(const std::shared_ptr<const CuMatrix<double>>& src) { _pimpl->copy_from(src); }
void CuMatrix<complex>::copy_from(const std::shared_ptr<const CuMatrix<complex>>& src) { _pimpl->copy_from(src); }
void CuMatrix<double>::copy_from(const double* v, const size_t n) { _pimpl->copy_from(v, n); }
void CuMatrix<complex>::copy_from(const complex* v, const size_t n) { _pimpl->copy_from(v, n); }
void CuMatrix<double>::copy_to_host() { _pimpl->copy_to_host(); }
void CuMatrix<complex>::copy_to_host() { _pimpl->copy_to_host(); }

void CuMatrix<double>::transfer_matrix(const double* foci, size_t foci_num, const std::vector<const double*>& positions,
                                       const std::vector<const double*>& directions, double wavelength, double attenuation) {}
void CuMatrix<complex>::transfer_matrix(const double* foci, size_t foci_num, const std::vector<const double*>& positions,
                                        const std::vector<const double*>& directions, double wavelength, double attenuation) {
  _pimpl->transfer_matrix(foci, foci_num, positions, directions, wavelength, attenuation);
}
void CuMatrix<double>::set_bcd_result(const std::shared_ptr<const CuMatrix<double>>& vec, size_t index) {}
void CuMatrix<complex>::set_bcd_result(const std::shared_ptr<const CuMatrix<complex>>& vec, size_t index) { _pimpl->set_bcd_result(vec, index); }
void CuMatrix<double>::set_from_complex_drive(std::vector<core::DataArray>& dst, bool normalize, double max_coefficient) {}
void CuMatrix<complex>::set_from_complex_drive(std::vector<core::DataArray>& dst, bool normalize, double max_coefficient) {
  _pimpl->set_from_complex_drive(dst, normalize, max_coefficient);
}
void CuMatrix<double>::set_from_arg(std::vector<core::DataArray>& dst, size_t n) { _pimpl->set_from_arg(dst, n); }
void CuMatrix<complex>::set_from_arg(std::vector<core::DataArray>& dst, size_t n) {}
void CuMatrix<double>::back_prop(const std::shared_ptr<const CuMatrix<double>>& transfer, const std::shared_ptr<const CuMatrix<double>>& amps) {}
void CuMatrix<complex>::back_prop(const std::shared_ptr<const CuMatrix<complex>>& transfer, const std::shared_ptr<const CuMatrix<complex>>& amps) {
  _pimpl->back_prop(transfer, amps);
}
void CuMatrix<double>::sigma_regularization(const std::shared_ptr<const CuMatrix<double>>& transfer,
                                            const std::shared_ptr<const CuMatrix<double>>& amps, double gamma) {}
void CuMatrix<complex>::sigma_regularization(const std::shared_ptr<const CuMatrix<complex>>& transfer,
                                             const std::shared_ptr<const CuMatrix<complex>>& amps, double gamma) {
  _pimpl->sigma_regularization(transfer, amps, gamma);
}
void CuMatrix<double>::col_sum_imag(const std::shared_ptr<CuMatrix<complex>>& src) { _pimpl->col_sum_imag(src); }
void CuMatrix<complex>::col_sum_imag(const std::shared_ptr<CuMatrix<complex>>& src) {}

}  // namespace holo
}  // namespace gain
}  // namespace autd
