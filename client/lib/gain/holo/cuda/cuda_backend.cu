// File: cuda_backend.cpp
// Project: cuda
// Created Date: 04/09/2021
// Author: Shun Suzuki
// -----
// Last Modified: 04/09/2021
// Modified By: Shun Suzuki (suzuki@hapis.k.u-tokyo.ac.jp)
// -----
// Copyright (c) 2021 Hapis Lab. All rights reserved.
//

#include "autd3/gain/cuda_backend.hpp"

namespace autd {
namespace gain {
namespace holo {

CUDABackend::CUDABackend() { cublasCreate_v2(&_handle); }

CUDABackend::~CUDABackend() { cublasDestroy_v2(_handle); }

BackendPtr CUDABackend::create() { return std::make_shared<CUDABackend>(); }

bool CUDABackend::supports_svd() { return true; }
bool CUDABackend::supports_evd() { return true; }
bool CUDABackend::supports_solve() { return true; }

void CUDABackend::hadamard_product(const MatrixXc& a, const MatrixXc& b, MatrixXc* c) {}
void CUDABackend::real(const MatrixXc& a, MatrixX* b) {}
void CUDABackend::pseudo_inverse_svd(const MatrixXc& matrix, const double alpha, MatrixXc* result) {}

CUDABackend::VectorXc CUDABackend::max_eigen_vector(MatrixXc* matrix) { return matrix->col(0); }

void CUDABackend::matrix_add(const double alpha, const MatrixX& a, MatrixX* b) {}

void CUDABackend::matrix_mul(const TRANSPOSE trans_a, const TRANSPOSE trans_b, const std::complex<double> alpha, const MatrixXc& a, const MatrixXc& b,
                             const std::complex<double> beta, MatrixXc* c) {}

void CUDABackend::matrix_vector_mul(const TRANSPOSE trans_a, const std::complex<double> alpha, const MatrixXc& a, const VectorXc& b,
                                    const std::complex<double> beta, VectorXc* c) {}

void CUDABackend::vector_add(const double alpha, const VectorX& a, VectorX* b) {}

void CUDABackend::solve_g(MatrixX* a, VectorX* b, VectorX* c) {}

void CUDABackend::solve_ch(MatrixXc* a, VectorXc* b) {}

double CUDABackend::dot(const VectorX& a, const VectorX& b) { return 0.0; }
std::complex<double> CUDABackend::dot_c(const VectorXc& a, const VectorXc& b) { return 0.0; }

double CUDABackend::max_coefficient_c(const VectorXc& v) { return 0.0; }
double CUDABackend::max_coefficient(const VectorX& v) { return 0.0; }

CUDABackend::MatrixXc CUDABackend::concat_row(const MatrixXc& a, const MatrixXc& b) {
  MatrixXc c(a.rows() + b.rows(), b.cols());
  c << a, b;
  return c;
}

CUDABackend::MatrixXc CUDABackend::concat_col(const MatrixXc& a, const MatrixXc& b) {
  MatrixXc c(a.rows(), a.cols() + b.cols());
  c << a, b;
  return c;
}

void CUDABackend::mat_cpy(const MatrixX& a, MatrixX* b) {}

void CUDABackend::vec_cpy(const VectorX& a, VectorX* b) {}

void CUDABackend::vec_cpy_c(const VectorXc& a, VectorXc* b) {}

}  // namespace holo
}  // namespace gain
}  // namespace autd