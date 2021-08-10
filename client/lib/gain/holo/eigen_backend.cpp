// File: eigen_backend.cpp
// Project: holo_gain
// Created Date: 16/05/2021
// Author: Shun Suzuki
// -----
// Last Modified: 10/08/2021
// Modified By: Shun Suzuki (suzuki@hapis.k.u-tokyo.ac.jp)
// -----
// Copyright (c) 2021 Hapis Lab. All rights reserved.
//

#include "autd3/gain/eigen_backend.hpp"

namespace autd::gain::holo {
BackendPtr Eigen3Backend::create() { return std::make_shared<Eigen3Backend>(); }

bool Eigen3Backend::supports_svd() { return true; }
bool Eigen3Backend::supports_evd() { return true; }
bool Eigen3Backend::supports_solve() { return true; }

void Eigen3Backend::hadamard_product(const MatrixXc& a, const MatrixXc& b, MatrixXc* c) { (*c).noalias() = a.cwiseProduct(b); }
void Eigen3Backend::real(const MatrixXc& a, MatrixX* b) { (*b).noalias() = a.real(); }
void Eigen3Backend::pseudo_inverse_svd(MatrixXc* matrix, const double alpha, MatrixXc* result) {
  const Eigen::BDCSVD svd(*matrix, Eigen::ComputeThinU | Eigen::ComputeThinV);
  auto singular_values_inv = svd.singularValues();
  const auto size = singular_values_inv.size();
  for (Eigen::Index i = 0; i < size; i++) singular_values_inv(i) = singular_values_inv(i) / (singular_values_inv(i) * singular_values_inv(i) + alpha);
  (*result).noalias() = svd.matrixV() * singular_values_inv.asDiagonal() * svd.matrixU().adjoint();
}
Eigen3Backend::VectorXc Eigen3Backend::max_eigen_vector(MatrixXc* matrix) {
  const Eigen::ComplexEigenSolver<MatrixXc> ces(*matrix);
  auto idx = 0;
  ces.eigenvalues().cwiseAbs2().maxCoeff(&idx);
  return ces.eigenvectors().col(idx);
}
void Eigen3Backend::matrix_add(const double alpha, const MatrixX& a, const double beta, MatrixX* b) {
  *b *= beta;
  (*b).noalias() += alpha * a;
}
void Eigen3Backend::matrix_mul(const TRANSPOSE trans_a, const TRANSPOSE trans_b, const std::complex<double> alpha, const MatrixXc& a,
                               const MatrixXc& b, const std::complex<double> beta, MatrixXc* c) {
  *c *= beta;
  switch (trans_a) {
    case TRANSPOSE::CONJ_TRANS:
      switch (trans_b) {
        case TRANSPOSE::CONJ_TRANS:
          (*c).noalias() += alpha * (a.adjoint() * b.adjoint());
          break;
        case TRANSPOSE::TRANS:
          (*c).noalias() += alpha * (a.adjoint() * b.transpose());
          break;
        case TRANSPOSE::CONJ_NO_TRANS:
          (*c).noalias() += alpha * (a.adjoint() * b.conjugate());
          break;
        case TRANSPOSE::NO_TRANS:
          (*c).noalias() += alpha * (a.adjoint() * b);
          break;
      }
      break;
    case TRANSPOSE::TRANS:
      switch (trans_b) {
        case TRANSPOSE::CONJ_TRANS:
          (*c).noalias() += alpha * (a.transpose() * b.adjoint());
          break;
        case TRANSPOSE::TRANS:
          (*c).noalias() += alpha * (a.transpose() * b.transpose());
          break;
        case TRANSPOSE::CONJ_NO_TRANS:
          (*c).noalias() += alpha * (a.transpose() * b.conjugate());
          break;
        case TRANSPOSE::NO_TRANS:
          (*c).noalias() += alpha * (a.transpose() * b);
          break;
      }
      break;
    case TRANSPOSE::CONJ_NO_TRANS:
      switch (trans_b) {
        case TRANSPOSE::CONJ_TRANS:
          (*c).noalias() += alpha * (a.conjugate() * b.adjoint());
          break;
        case TRANSPOSE::TRANS:
          (*c).noalias() += alpha * (a.conjugate() * b.transpose());
          break;
        case TRANSPOSE::CONJ_NO_TRANS:
          (*c).noalias() += alpha * (a.conjugate() * b.conjugate());
          break;
        case TRANSPOSE::NO_TRANS:
          (*c).noalias() += alpha * (a.conjugate() * b);
          break;
      }
      break;
    case TRANSPOSE::NO_TRANS:
      switch (trans_b) {
        case TRANSPOSE::CONJ_TRANS:
          (*c).noalias() += alpha * (a * b.adjoint());
          break;
        case TRANSPOSE::TRANS:
          (*c).noalias() += alpha * (a * b.transpose());
          break;
        case TRANSPOSE::CONJ_NO_TRANS:
          (*c).noalias() += alpha * (a * b.conjugate());
          break;
        case TRANSPOSE::NO_TRANS:
          (*c).noalias() += alpha * (a * b);
          break;
      }
      break;
  }
}
void Eigen3Backend::matrix_vector_mul(const TRANSPOSE trans_a, const std::complex<double> alpha, const MatrixXc& a, const VectorXc& b,
                                      const std::complex<double> beta, VectorXc* c) {
  *c *= beta;
  switch (trans_a) {
    case TRANSPOSE::CONJ_TRANS:
      (*c).noalias() += alpha * (a.adjoint() * b);
      break;
    case TRANSPOSE::TRANS:
      (*c).noalias() += alpha * (a.transpose() * b);
      break;
    case TRANSPOSE::CONJ_NO_TRANS:
      (*c).noalias() += alpha * (a.conjugate() * b);
      break;
    case TRANSPOSE::NO_TRANS:
      (*c).noalias() += alpha * (a * b);
      break;
  }
}
void Eigen3Backend::vector_add(const double alpha, const VectorX& a, const double beta, VectorX* b) {
  *b *= beta;
  (*b).noalias() += alpha * a;
}
void Eigen3Backend::solve_g(MatrixX* a, VectorX* b, VectorX* c) {
  const Eigen::HouseholderQR<MatrixX> qr(*a);
  (*c).noalias() = qr.solve(*b);
}
void Eigen3Backend::solve_ch(MatrixXc* a, VectorXc* b) {
  const Eigen::LLT<MatrixXc> llt(*a);
  llt.solveInPlace(*b);
}
double Eigen3Backend::dot(const VectorX& a, const VectorX& b) { return a.dot(b); }
std::complex<double> Eigen3Backend::dot_c(const VectorXc& a, const VectorXc& b) { return a.conjugate().dot(b); }
double Eigen3Backend::max_coefficient_c(const VectorXc& v) { return sqrt(v.cwiseAbs2().maxCoeff()); }
double Eigen3Backend::max_coefficient(const VectorX& v) { return v.maxCoeff(); }
Eigen3Backend::MatrixXc Eigen3Backend::concat_row(const MatrixXc& a, const MatrixXc& b) {
  MatrixXc c(a.rows() + b.rows(), b.cols());
  c << a, b;
  return c;
}
Eigen3Backend::MatrixXc Eigen3Backend::concat_col(const MatrixXc& a, const MatrixXc& b) {
  MatrixXc c(a.rows(), a.cols() + b.cols());
  c << a, b;
  return c;
}
void Eigen3Backend::mat_cpy(const MatrixX& a, MatrixX* b) { *b = a; }
void Eigen3Backend::vec_cpy(const VectorX& a, VectorX* b) { *b = a; }
void Eigen3Backend::vec_cpy_c(const VectorXc& a, VectorXc* b) { *b = a; }

}  // namespace autd::gain::holo
