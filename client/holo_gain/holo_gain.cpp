// File: holo_gain.cpp
// Project: holo_gain
// Created Date: 16/05/2021
// Author: Shun Suzuki
// -----
// Last Modified: 16/05/2021
// Modified By: Shun Suzuki (suzuki@hapis.k.u-tokyo.ac.jp)
// -----
// Copyright (c) 2021 Hapis Lab. All rights reserved.
//

#include "holo_gain.hpp"

#include "linalg_backend.hpp"

namespace autd::gain::holo {

namespace {
void SetBcdResult(Backend::MatrixXc& mat, const Backend::VectorXc& vec, const size_t idx) {
  const size_t m = vec.size();
  for (size_t i = 0; i < idx; i++) mat(idx, i) = std::conj(vec(i));
  for (auto i = idx + 1; i < m; i++) mat(idx, i) = std::conj(vec(i));
  for (size_t i = 0; i < idx; i++) mat(i, idx) = vec(i);
  for (auto i = idx + 1; i < m; i++) mat(i, idx) = vec(i);
}
}  // namespace

Result<bool, std::string> HoloGainSDP::Calc(const core::GeometryPtr& geometry) {
  if (!this->_backend->SupportsSvd() || !this->_backend->SupportsEVD()) return Err(std::string("This backend does not support this method."));

  const auto m = this->_foci.size();
  const auto n = geometry->num_transducers();

  Backend::MatrixXc p = Backend::MatrixXc::Zero(m, m);
  for (size_t i = 0; i < m; i++) p(i, i) = std::complex<double>(this->_amps[i], 0);

  auto b = this->TransferMatrix(geometry);
  Backend::MatrixXc pseudo_inv_b(n, m);
  this->_backend->PseudoInverseSVD(&b, _alpha, &pseudo_inv_b);

  Backend::MatrixXc mm = Backend::MatrixXc::Identity(m, m);
  this->_backend->MatMul(TRANSPOSE::NO_TRANS, TRANSPOSE::NO_TRANS, std::complex<double>(1, 0), b, pseudo_inv_b, std::complex<double>(-1, 0), &mm);
  Backend::MatrixXc tmp = Backend::MatrixXc::Zero(m, m);
  this->MatrixMul(p, mm, &tmp);
  this->MatrixMul(tmp, p, &mm);
  Backend::MatrixXc x_mat = Backend::MatrixXc::Identity(m, m);

  std::random_device rnd;
  std::mt19937 mt(rnd());
  std::uniform_real_distribution<double> range(0, 1);
  const Backend::VectorXc zero = Backend::VectorXc::Zero(m);
  Backend::VectorXc x = Backend::VectorXc::Zero(m);
  for (size_t i = 0; i < _repeat; i++) {
    const auto ii = static_cast<size_t>(static_cast<double>(m) * range(mt));

    Backend::VectorXc mmc = mm.col(ii);
    mmc(ii) = 0;

    this->MatrixVecMul(x_mat, mmc, &x);
    if (std::complex<double> gamma = this->_backend->DotC(x, mmc); gamma.real() > 0) {
      x = -x * sqrt(_lambda / gamma.real());
      SetBcdResult(x_mat, x, ii);
    } else {
      SetBcdResult(x_mat, zero, ii);
    }
  }

  const Backend::VectorXc u = this->_backend->MaxEigenVector(&x_mat);

  Backend::VectorXc ut = Backend::VectorXc::Zero(m);
  this->MatrixVecMul(p, u, &ut);

  Backend::VectorXc q = Backend::VectorXc::Zero(n);
  this->MatrixVecMul(pseudo_inv_b, ut, &q);

  const auto max_coefficient = this->_backend->MaxCoeffC(q);
  this->SetFromComplexDrive(this->_data, q, _normalize, max_coefficient);

  this->_built = true;
  return Ok(true);
}

}  // namespace autd::gain::holo