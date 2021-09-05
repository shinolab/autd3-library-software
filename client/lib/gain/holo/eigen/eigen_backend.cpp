// File: eigen_backend.cpp
// Project: holo_gain
// Created Date: 16/05/2021
// Author: Shun Suzuki
// -----
// Last Modified: 05/09/2021
// Modified By: Shun Suzuki (suzuki@hapis.k.u-tokyo.ac.jp)
// -----
// Copyright (c) 2021 Hapis Lab. All rights reserved.
//

#include "autd3/gain/eigen_backend.hpp"

#include "autd3/core/utils.hpp"
#include "autd3/utils.hpp"

namespace autd::gain::holo {
BackendPtr Eigen3Backend::create() { return std::make_shared<Eigen3Backend>(); }

void Eigen3Backend::hadamard_product(const std::shared_ptr<MatrixXc> a, const std::shared_ptr<MatrixXc> b, std::shared_ptr<MatrixXc> c) {
  c->data.noalias() = a->data.cwiseProduct(b->data);
}
void Eigen3Backend::hadamard_product(const std::shared_ptr<VectorXc> a, const std::shared_ptr<VectorXc> b, std::shared_ptr<VectorXc> c) {
  c->data.noalias() = a->data.cwiseProduct(b->data);
}
void Eigen3Backend::real(const std::shared_ptr<MatrixXc> a, std::shared_ptr<MatrixX> b) { b->data.noalias() = a->data.real(); }
void Eigen3Backend::arg(const std::shared_ptr<VectorXc> a, std::shared_ptr<VectorXc> c) { c->data = a->data.cwiseQuotient(a->data.cwiseAbs()); }
void Eigen3Backend::pseudo_inverse_svd(const std::shared_ptr<MatrixXc> matrix, const double alpha, std::shared_ptr<MatrixXc> result) {
  const Eigen::BDCSVD svd(matrix->data, Eigen::ComputeFullU | Eigen::ComputeFullV);
  auto singular_values_inv = svd.singularValues();
  const auto size = singular_values_inv.size();
  for (Eigen::Index i = 0; i < size; i++) singular_values_inv(i) = singular_values_inv(i) / (singular_values_inv(i) * singular_values_inv(i) + alpha);

  result->data.noalias() = svd.matrixV() * singular_values_inv.asDiagonal() * svd.matrixU().adjoint();
}
std::shared_ptr<VectorXc> Eigen3Backend::max_eigen_vector(std::shared_ptr<MatrixXc> matrix) {
  const Eigen::ComplexEigenSolver<Eigen::Matrix<complex, -1, -1, Eigen::ColMajor>> ces(matrix->data);
  auto idx = 0;
  ces.eigenvalues().cwiseAbs2().maxCoeff(&idx);
  return std::make_shared<VectorXc>(ces.eigenvectors().col(idx));
}

void Eigen3Backend::matrix_add(const double alpha, const std::shared_ptr<MatrixX> a, std::shared_ptr<MatrixX> b) {
  b->data.noalias() += alpha * a->data;
}
void Eigen3Backend::matrix_mul(const TRANSPOSE trans_a, const TRANSPOSE trans_b, const complex alpha, const std::shared_ptr<MatrixXc> a,
                               const std::shared_ptr<MatrixXc> b, const complex beta, std::shared_ptr<MatrixXc> c) {
  c->data *= beta;
  switch (trans_a) {
    case TRANSPOSE::CONJ_TRANS:
      switch (trans_b) {
        case TRANSPOSE::CONJ_TRANS:
          c->data.noalias() += alpha * (a->data.adjoint() * b->data.adjoint());
          break;
        case TRANSPOSE::TRANS:
          c->data.noalias() += alpha * (a->data.adjoint() * b->data.transpose());
          break;
        case TRANSPOSE::CONJ_NO_TRANS:
          c->data.noalias() += alpha * (a->data.adjoint() * b->data.conjugate());
          break;
        case TRANSPOSE::NO_TRANS:
          c->data.noalias() += alpha * (a->data.adjoint() * b->data);
          break;
      }
      break;
    case TRANSPOSE::TRANS:
      switch (trans_b) {
        case TRANSPOSE::CONJ_TRANS:
          c->data.noalias() += alpha * (a->data.transpose() * b->data.adjoint());
          break;
        case TRANSPOSE::TRANS:
          c->data.noalias() += alpha * (a->data.transpose() * b->data.transpose());
          break;
        case TRANSPOSE::CONJ_NO_TRANS:
          c->data.noalias() += alpha * (a->data.transpose() * b->data.conjugate());
          break;
        case TRANSPOSE::NO_TRANS:
          c->data.noalias() += alpha * (a->data.transpose() * b->data);
          break;
      }
      break;
    case TRANSPOSE::CONJ_NO_TRANS:
      switch (trans_b) {
        case TRANSPOSE::CONJ_TRANS:
          c->data.noalias() += alpha * (a->data.conjugate() * b->data.adjoint());
          break;
        case TRANSPOSE::TRANS:
          c->data.noalias() += alpha * (a->data.conjugate() * b->data.transpose());
          break;
        case TRANSPOSE::CONJ_NO_TRANS:
          c->data.noalias() += alpha * (a->data.conjugate() * b->data.conjugate());
          break;
        case TRANSPOSE::NO_TRANS:
          c->data.noalias() += alpha * (a->data.conjugate() * b->data);
          break;
      }
      break;
    case TRANSPOSE::NO_TRANS:
      switch (trans_b) {
        case TRANSPOSE::CONJ_TRANS:
          c->data.noalias() += alpha * (a->data * b->data.adjoint());
          break;
        case TRANSPOSE::TRANS:
          c->data.noalias() += alpha * (a->data * b->data.transpose());
          break;
        case TRANSPOSE::CONJ_NO_TRANS:
          c->data.noalias() += alpha * (a->data * b->data.conjugate());
          break;
        case TRANSPOSE::NO_TRANS:
          c->data.noalias() += alpha * (a->data * b->data);
          break;
      }
      break;
  }
}
void Eigen3Backend::matrix_mul(const TRANSPOSE trans_a, const TRANSPOSE trans_b, const double alpha, const std::shared_ptr<MatrixX> a,
                               const std::shared_ptr<MatrixX> b, const double beta, std::shared_ptr<MatrixX> c) {
  c->data *= beta;
  switch (trans_a) {
    case TRANSPOSE::TRANS:
      switch (trans_b) {
        case TRANSPOSE::TRANS:
          c->data.noalias() += alpha * (a->data.transpose() * b->data.transpose());
          break;
        case TRANSPOSE::NO_TRANS:
          c->data.noalias() += alpha * (a->data.transpose() * b->data);
          break;
        default:
          throw std::runtime_error("invalid operation");
      }
      break;
    case TRANSPOSE::NO_TRANS:
      switch (trans_b) {
        case TRANSPOSE::TRANS:
          c->data.noalias() += alpha * (a->data * b->data.transpose());
          break;
        case TRANSPOSE::NO_TRANS:
          c->data.noalias() += alpha * (a->data * b->data);
          break;
        default:
          throw std::runtime_error("invalid operation");
      }
      break;
    default:
      throw std::runtime_error("invalid operation");
  }
}
void Eigen3Backend::matrix_vector_mul(const TRANSPOSE trans_a, const complex alpha, const std::shared_ptr<MatrixXc> a,
                                      const std::shared_ptr<VectorXc> b, const complex beta, std::shared_ptr<VectorXc> c) {
  c->data *= beta;
  switch (trans_a) {
    case TRANSPOSE::CONJ_TRANS:
      c->data.noalias() += alpha * (a->data.adjoint() * b->data);
      break;
    case TRANSPOSE::TRANS:
      c->data.noalias() += alpha * (a->data.transpose() * b->data);
      break;
    case TRANSPOSE::CONJ_NO_TRANS:
      c->data.noalias() += alpha * (a->data.conjugate() * b->data);
      break;
    case TRANSPOSE::NO_TRANS:
      c->data.noalias() += alpha * (a->data * b->data);
      break;
  }
}

void Eigen3Backend::matrix_vector_mul(const TRANSPOSE trans_a, const double alpha, const std::shared_ptr<MatrixX> a, const std::shared_ptr<VectorX> b,
                                      const double beta, std::shared_ptr<VectorX> c) {
  c->data *= beta;
  switch (trans_a) {
    case TRANSPOSE::TRANS:
      c->data.noalias() += alpha * (a->data.transpose() * b->data);
      break;
    case TRANSPOSE::NO_TRANS:
      c->data.noalias() += alpha * (a->data * b->data);
      break;
    default:
      throw std::runtime_error("invalid operation");
  }
}

void Eigen3Backend::vector_add(const double alpha, const std::shared_ptr<VectorX> a, std::shared_ptr<VectorX> b) {
  b->data.noalias() += alpha * a->data;
}
void Eigen3Backend::solve_g(std::shared_ptr<MatrixX> a, std::shared_ptr<VectorX> b, std::shared_ptr<VectorX> c) {
  const Eigen::HouseholderQR<Eigen::Matrix<double, -1, -1, Eigen::ColMajor>> qr(a->data);
  c->data.noalias() = qr.solve(b->data);
}
void Eigen3Backend::solve_ch(std::shared_ptr<MatrixXc> a, std::shared_ptr<VectorXc> b) {
  const Eigen::LLT<Eigen::Matrix<complex, -1, -1, Eigen::ColMajor>> llt(a->data);
  llt.solveInPlace(b->data);
}
double Eigen3Backend::dot(const std::shared_ptr<VectorX> a, const std::shared_ptr<VectorX> b) { return a->data.dot(b->data); }
complex Eigen3Backend::dot(const std::shared_ptr<VectorXc> a, const std::shared_ptr<VectorXc> b) { return a->data.conjugate().dot(b->data); }
double Eigen3Backend::max_coefficient(const std::shared_ptr<VectorXc> v) { return sqrt(v->data.cwiseAbs2().maxCoeff()); }
double Eigen3Backend::max_coefficient(const std::shared_ptr<VectorX> v) { return v->data.maxCoeff(); }
std::shared_ptr<MatrixXc> Eigen3Backend::concat_row(const std::shared_ptr<MatrixXc> a, const std::shared_ptr<MatrixXc> b) {
  Eigen::Matrix<complex, -1, -1, Eigen::ColMajor> c(a->data.rows() + b->data.rows(), b->data.cols());
  c << a->data, b->data;
  return std::make_shared<MatrixXc>(c);
}
std::shared_ptr<MatrixXc> Eigen3Backend::concat_col(const std::shared_ptr<MatrixXc> a, const std::shared_ptr<MatrixXc> b) {
  Eigen::Matrix<complex, -1, -1, Eigen::ColMajor> c(a->data.rows(), a->data.cols() + b->data.cols());
  c << a->data, b->data;
  return std::make_shared<MatrixXc>(c);
}
void Eigen3Backend::mat_cpy(const std::shared_ptr<MatrixX> a, std::shared_ptr<MatrixX> b) { b->data = a->data; }
void Eigen3Backend::vec_cpy(const std::shared_ptr<VectorX> a, std::shared_ptr<VectorX> b) { b->data = a->data; }
void Eigen3Backend::vec_cpy(const std::shared_ptr<VectorXc> a, std::shared_ptr<VectorXc> b) { b->data = a->data; }

void Eigen3Backend::set_from_complex_drive(std::vector<core::DataArray>& data, const std::shared_ptr<VectorXc> drive, const bool normalize,
                                           const double max_coefficient) {
  const Eigen::Index n = drive->data.size();
  size_t dev_idx = 0;
  size_t trans_idx = 0;
  for (Eigen::Index j = 0; j < n; j++) {
    const auto f_amp = normalize ? 1.0 : std::abs(drive->data(j)) / max_coefficient;
    const auto f_phase = std::arg(drive->data(j)) / (2.0 * M_PI);
    const auto phase = core::Utilities::to_phase(f_phase);
    const auto duty = core::Utilities::to_duty(f_amp);
    data[dev_idx][trans_idx++] = core::Utilities::pack_to_u16(duty, phase);
    if (trans_idx == core::NUM_TRANS_IN_UNIT) {
      dev_idx++;
      trans_idx = 0;
    }
  }
}

complex transfer(const core::Vector3& trans_pos, const core::Vector3& trans_norm, const core::Vector3& target_pos, const double wave_number,
                 const double attenuation) {
  const auto diff = target_pos - trans_pos;
  const auto dist = diff.norm();
  const auto theta = std::atan2(diff.dot(trans_norm), dist * trans_norm.norm()) * 180.0 / M_PI;
  const auto directivity = utils::Directivity::t4010a1(theta);
  return directivity / dist * exp(complex(-dist * attenuation, -wave_number * dist));
}

std::shared_ptr<MatrixXc> Eigen3Backend::transfer_matrix(const std::vector<core::Vector3>& foci, const core::GeometryPtr& geometry) {
  const auto m = static_cast<Eigen::Index>(foci.size());
  const auto n = static_cast<Eigen::Index>(geometry->num_transducers());

  auto g = allocate_matrix_c("g", m, n);

  const auto wave_number = 2.0 * M_PI / geometry->wavelength();
  const auto attenuation = geometry->attenuation_coefficient();
  for (Eigen::Index i = 0; i < m; i++) {
    const auto& tp = foci[i];
    for (Eigen::Index j = 0; j < n; j++) {
      const auto pos = geometry->position(j);
      const auto dir = geometry->direction(j / core::NUM_TRANS_IN_UNIT);
      g->data(i, j) = transfer(pos, dir, tp, wave_number, attenuation);
    }
  }
  return g;
}

}  // namespace autd::gain::holo
