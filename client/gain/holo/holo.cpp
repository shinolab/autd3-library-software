// File: holo_gain.cpp
// Project: lib
// Created Date: 06/07/2016
// Author: Seki Inoue
// -----
// Last Modified: 01/03/2021
// Modified By: Shun Suzuki (suzuki@hapis.k.u-tokyo.ac.jp)
// -----
// Copyright (c) 2016-2020 Hapis Lab. All rights reserved.
//

#include "gain/holo.hpp"

namespace autd::gain {

void Eigen3Backend::hadamardProduct(const Eigen3Backend::MatrixXc& a, const Eigen3Backend::MatrixXc& b, Eigen3Backend::MatrixXc* c) {
  (*c).noalias() = a.cwiseProduct(b);
}
void Eigen3Backend::real(const Eigen3Backend::MatrixXc& a, Eigen3Backend::MatrixX* b) { (*b).noalias() = a.real(); }
Eigen3Backend::MatrixXc Eigen3Backend::pseudoInverseSVD(const Eigen3Backend::MatrixXc& matrix, Float alpha) {
  Eigen::JacobiSVD<MatrixXc> svd(matrix, Eigen::ComputeThinU | Eigen::ComputeThinV);
  Eigen::JacobiSVD<MatrixXc>::SingularValuesType singularValues_inv = svd.singularValues();
  for (auto i = 0; i < singularValues_inv.size(); i++) {
    singularValues_inv(i) = singularValues_inv(i) / (singularValues_inv(i) * singularValues_inv(i) + alpha);
  }
  MatrixXc pinvB = (svd.matrixV() * singularValues_inv.asDiagonal() * svd.matrixU().adjoint());
  return pinvB;
}

Eigen3Backend::VectorXc Eigen3Backend::maxEigenVector(const Eigen3Backend::MatrixXc& matrix) {
  const Eigen::ComplexEigenSolver<MatrixXc> ces(matrix);
  int idx = 0;
  ces.eigenvalues().cwiseAbs2().maxCoeff(&idx);
  return ces.eigenvectors().col(idx);
}

void Eigen3Backend::matadd(Float alpha, const MatrixX& a, Float beta, MatrixX* b) {
  *b *= beta;
  (*b).noalias() += alpha * a;
}

void Eigen3Backend::matmul(const char* transa, const char* transb, std::complex<Float> alpha, const Eigen3Backend::MatrixXc& a,
                           const Eigen3Backend::MatrixXc& b, std::complex<Float> beta, Eigen3Backend::MatrixXc* c) {
  *c *= beta;
  if (strcmp(transa, "C") == 0) {
    if (strcmp(transb, "C") == 0) {
      (*c).noalias() += alpha * (a.adjoint() * b.adjoint());
    } else if (strcmp(transb, "T") == 0) {
      (*c).noalias() += alpha * (a.adjoint() * b.transpose());
    } else {
      (*c).noalias() += alpha * (a.adjoint() * b);
    }
  } else if (strcmp(transa, "T") == 0) {
    if (strcmp(transb, "C") == 0) {
      (*c).noalias() += alpha * (a.transpose() * b.adjoint());
    } else if (strcmp(transb, "T") == 0) {
      (*c).noalias() += alpha * (a.transpose() * b.transpose());
    } else {
      (*c).noalias() += alpha * (a.transpose() * b);
    }
  } else {
    if (strcmp(transb, "C") == 0) {
      (*c).noalias() += alpha * (a * b.adjoint());
    } else if (strcmp(transb, "T") == 0) {
      (*c).noalias() += alpha * (a * b.transpose());
    } else {
      (*c).noalias() += alpha * (a * b);
    }
  }
}

void Eigen3Backend::matvecmul(const char* transa, std::complex<Float> alpha, const Eigen3Backend::MatrixXc& a, const Eigen3Backend::VectorXc& b,
                              std::complex<Float> beta, Eigen3Backend::VectorXc* c) {
  *c *= beta;
  if (strcmp(transa, "C") == 0) {
    (*c).noalias() += alpha * (a.adjoint() * b);
  } else if (strcmp(transa, "T") == 0) {
    (*c).noalias() += alpha * (a.transpose() * b);
  } else {
    (*c).noalias() += alpha * (a * b);
  }
}

void Eigen3Backend::vecadd(Float alpha, const Eigen3Backend::VectorX& a, Float beta, Eigen3Backend::VectorX* b) {
  *b *= beta;
  (*b).noalias() += alpha * a;
}
void Eigen3Backend::solve(const Eigen3Backend::MatrixX& a, const Eigen3Backend::VectorX& b, Eigen3Backend::VectorX* c) {
  Eigen::HouseholderQR<Eigen3Backend::MatrixX> qr(a);
  (*c).noalias() = qr.solve(b);
}

void Eigen3Backend::csolve(const Eigen3Backend::MatrixXc& a, const Eigen3Backend::VectorXc& b, Eigen3Backend::VectorXc* c) {
  Eigen::FullPivHouseholderQR<Eigen3Backend::MatrixXc> qr(a);
  (*c).noalias() = qr.solve(b);
}

Float Eigen3Backend::dot(const Eigen3Backend::VectorX& a, const Eigen3Backend::VectorX& b) { return a.dot(b); }

std::complex<Float> Eigen3Backend::cdot(const Eigen3Backend::VectorXc& a, const Eigen3Backend::VectorXc& b) { return a.conjugate().dot(b); }

Float Eigen3Backend::maxCoeff(const Eigen3Backend::VectorXc& v) { return sqrt(v.cwiseAbs2().maxCoeff()); }

Eigen3Backend::MatrixXc Eigen3Backend::concat_in_row(const Eigen3Backend::MatrixXc& a, const Eigen3Backend::MatrixXc& b) {
  Eigen3Backend::MatrixXc c(a.rows() + b.rows(), b.cols());
  c << a, b;
  return c;
}
Eigen3Backend::MatrixXc Eigen3Backend::concat_in_col(const Eigen3Backend::MatrixXc& a, const Eigen3Backend::MatrixXc& b) {
  Eigen3Backend::MatrixXc c(a.rows(), a.cols() + b.cols());
  c << a, b;
  return c;
}

}  // namespace autd::gain
