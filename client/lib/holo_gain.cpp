// File: holo_gain.cpp
// Project: lib
// Created Date: 06/07/2016
// Author: Seki Inoue
// -----
// Last Modified: 30/04/2020
// Modified By: Shun Suzuki (suzuki@hapis.k.u-tokyo.ac.jp)
// -----
// Copyright (c) 2016-2020 Hapis Lab. All rights reserved.
//

#define _USE_MATH_DEFINES
#include <math.h>

#include <complex>
#include <iostream>
#include <map>
#include <random>
#include <string>
#include <vector>

#if WIN32
#include <codeanalysis\warnings.h>
#pragma warning(push)
#pragma warning(disable : ALL_CODE_ANALYSIS_WARNINGS)
#endif
#include <Eigen/Eigen>
#if WIN32
#pragma warning(pop)
#endif

#include "consts.hpp"
#include "gain.hpp"
#include "privdef.hpp"

using autd::ULTRASOUND_WAVELENGTH;

static constexpr auto REPEAT_SDP = 10;
static constexpr auto LAMBDA_SDP = 0.8;
static constexpr auto ATTENUATION = 1.15e-4;

namespace hologainimpl {
using Eigen::MatrixXcd;
using Eigen::Vector3d;
using std::complex;

const double DIR_COEF_A[] = {1.0, 1.0, 1.0, 0.891250938, 0.707945784, 0.501187234, 0.354813389, 0.251188643, 0.199526231};
const double DIR_COEF_B[] = {
    0., 0., -0.00459648054721, -0.0155520765675, -0.0208114779827, -0.0182211227016, -0.0122437497109, -0.00780345575475, -0.00312857467007};
const double DIR_COEF_C[] = {
    0., 0., -0.000787968093807, -0.000307591508224, -0.000218348633296, 0.00047738416141, 0.000120353137658, 0.000323676257958, 0.000143850511};
const double DIR_COEF_D[] = {
    0., 0., 1.60125528528e-05, 2.9747624976e-06, 2.31910931569e-05, -1.1901034125e-05, 6.77743734332e-06, -5.99548024824e-06, -4.79372835035e-06};

static double directivity_t4010a1(double theta_deg) {
  theta_deg = abs(theta_deg);

  while (theta_deg > 90.0) {
    theta_deg = abs(180.0 - theta_deg);
  }

  size_t i = static_cast<size_t>(ceil(theta_deg / 10.0));

  if (i == 0) {
    return 1.0;
  } else {
    auto a = DIR_COEF_A[i - 1];
    auto b = DIR_COEF_B[i - 1];
    auto c = DIR_COEF_C[i - 1];
    auto d = DIR_COEF_D[i - 1];
    auto x = theta_deg - (i - 1.0) * 10.0;
    return a + b * x + c * x * x + d * x * x * x;
  }
}

complex<double> transfer(Eigen::Vector3d trans_pos, Eigen::Vector3d trans_norm, Eigen::Vector3d target_pos) {
  const auto diff = target_pos - trans_pos;
  const auto dist = diff.norm();
  const auto theta = atan2(diff.dot(trans_norm), dist * trans_norm.norm()) * 180.0 / M_PI;
  const auto directivity = directivity_t4010a1(theta);

  return directivity / dist * exp(complex<double>(-dist * ATTENUATION, -2 * M_PI / ULTRASOUND_WAVELENGTH * dist));
}

void removeRow(MatrixXcd* const matrix, size_t row_to_remove) {
  const auto num_rows = static_cast<size_t>(matrix->rows()) - 1;
  const auto num_cols = static_cast<size_t>(matrix->cols());

  if (row_to_remove < num_rows)
    matrix->block(row_to_remove, 0, num_rows - row_to_remove, num_cols) = matrix->block(row_to_remove + 1, 0, num_rows - row_to_remove, num_cols);

  matrix->conservativeResize(num_rows, num_cols);
}

void removeColumn(MatrixXcd* const matrix, size_t col_to_remove) {
  const auto num_rows = static_cast<size_t>(matrix->rows());
  const auto num_cols = static_cast<size_t>(matrix->cols()) - 1;

  if (col_to_remove < num_cols)
    matrix->block(0, col_to_remove, num_rows, num_cols - col_to_remove) = matrix->block(0, col_to_remove + 1, num_rows, num_cols - col_to_remove);

  matrix->conservativeResize(num_rows, num_cols);
}
}  // namespace hologainimpl

namespace autd {
namespace gain {

GainPtr HoloGainSdp::Create(std::vector<Vector3> foci, std::vector<double> amps) {
  auto ptr = CreateHelper<HoloGainSdp>();
  ptr->_foci = foci;
  ptr->_amps = amps;
  return ptr;
}

void HoloGainSdp::Build() {
  if (this->built()) return;
  auto geo = this->geometry();
  if (geo == nullptr) {
    throw std::runtime_error("Geometry is required to build Gain");
  }

  const auto alpha = 1e-3;

  const size_t M = _foci.size();
  Eigen::MatrixX3d foci(M, 3);
  Eigen::VectorXd amps(M);

  for (size_t i = 0; i < M; i++) {
    foci(i, 0) = _foci[i].x();
    foci(i, 1) = _foci[i].y();
    foci(i, 2) = _foci[i].z();
    amps(i) = _amps[i];
  }

  const auto N = static_cast<int>(geo->numTransducers());

  Eigen::MatrixXcd P = Eigen::MatrixXcd::Zero(M, M);
  Eigen::MatrixXcd B = Eigen::MatrixXcd(M, N);

  std::random_device rnd;
  std::mt19937 mt(rnd());
  std::uniform_real_distribution<double> range(0, 1);

  for (size_t i = 0; i < M; i++) {
    P(i, i) = amps(i);

    const auto tp = foci.row(i);
    for (int j = 0; j < N; j++) {
      const auto pos = geo->position(j);
      const auto dir = geo->direction(j);
      B(i, j) = hologainimpl::transfer(Eigen::Vector3d(pos.x(), pos.y(), pos.z()), Eigen::Vector3d(dir.x(), dir.y(), dir.z()), tp);
    }
  }

  Eigen::JacobiSVD<Eigen::MatrixXcd> svd(B, Eigen::ComputeThinU | Eigen::ComputeThinV);
  Eigen::JacobiSVD<Eigen::MatrixXcd>::SingularValuesType singularValues_inv = svd.singularValues();
  for (int64_t i = 0; i < singularValues_inv.size(); ++i) {
    singularValues_inv(i) = singularValues_inv(i) / (singularValues_inv(i) * singularValues_inv(i) + alpha * alpha);
  }
  Eigen::MatrixXcd pinvB = (svd.matrixV() * singularValues_inv.asDiagonal() * svd.matrixU().adjoint());

  Eigen::MatrixXcd MM = P * (Eigen::MatrixXcd::Identity(M, M) - B * pinvB) * P;
  Eigen::MatrixXcd X = Eigen::MatrixXcd::Identity(M, M);
  for (size_t i = 0; i < M * REPEAT_SDP; i++) {
    auto ii = static_cast<size_t>(M * static_cast<double>(range(mt)));

    auto Xc = X;
    hologainimpl::removeRow(&Xc, ii);
    hologainimpl::removeColumn(&Xc, ii);
    Eigen::VectorXcd MMc = MM.col(ii);
    MMc.block(ii, 0, MMc.rows() - 1 - ii, 1) = MMc.block(ii + 1, 0, MMc.rows() - 1 - ii, 1);
    MMc.conservativeResize(MMc.rows() - 1, 1);

    Eigen::VectorXcd x = Xc * MMc;
    std::complex<double> gamma = x.adjoint() * MMc;
    if (gamma.real() > 0) {
      x = -x * sqrt(LAMBDA_SDP / gamma.real());
      X.block(ii, 0, 1, ii) = x.block(0, 0, ii, 1).adjoint().eval();
      X.block(ii, ii + 1, 1, M - ii - 1) = x.block(ii, 0, M - 1 - ii, 1).adjoint().eval();
      X.block(0, ii, ii, 1) = x.block(0, 0, ii, 1).eval();
      X.block(ii + 1, ii, M - ii - 1, 1) = x.block(ii, 0, M - 1 - ii, 1).eval();
    } else {
      X.block(ii, 0, 1, ii) = Eigen::VectorXcd::Zero(ii).adjoint();
      X.block(ii, ii + 1, 1, M - ii - 1) = Eigen::VectorXcd::Zero(M - ii - 1).adjoint();
      X.block(0, ii, ii, 1) = Eigen::VectorXcd::Zero(ii);
      X.block(ii + 1, ii, M - ii - 1, 1) = Eigen::VectorXcd::Zero(M - ii - 1);
    }
  }

  Eigen::ComplexEigenSolver<Eigen::MatrixXcd> ces(X);
  Eigen::VectorXcd evs = ces.eigenvalues();
  double abseiv = 0;
  int idx = 0;
  for (int j = 0; j < evs.rows(); j++) {
    const auto eiv = abs(evs(j));
    if (abseiv < eiv) {
      abseiv = eiv;
      idx = j;
    }
  }

  Eigen::VectorXcd u = ces.eigenvectors().col(idx);
  const auto q = pinvB * P * u;

  this->_data.clear();
  const int ndevice = geo->numDevices();
  for (int i = 0; i < ndevice; i++) {
    this->_data[geo->deviceIdForDeviceIdx(i)].resize(NUM_TRANS_IN_UNIT);
  }

  // auto maxCoeff = sqrt(q.cwiseAbs2().maxCoeff());
  for (int j = 0; j < N; j++) {
    const auto famp = 1.0;  // abs(q(j)) / maxCoeff;
    const auto fphase = arg(q(j)) / (2 * M_PI) + 0.5;
    const auto amp = static_cast<uint8_t>(famp * 255);
    const auto phase = static_cast<uint8_t>((1 - fphase) * 255);
    uint8_t D, S;
    SignalDesign(amp, phase, &D, &S);
    this->_data[geo->deviceIdForTransIdx(j)].at(j % NUM_TRANS_IN_UNIT) = (static_cast<uint16_t>(D) << 8) + S;
  }
}
}  // namespace gain
}  // namespace autd