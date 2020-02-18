// File: holo_gain.cpp
// Project: lib
// Created Date: 06/07/2016
// Author: Seki Inoue
// -----
// Last Modified: 18/02/2020
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

#include "autd3.hpp"
#include "controller.hpp"
#include "gain.hpp"
#include "privdef.hpp"

static constexpr auto REPEAT_SDP = 10;
static constexpr auto LAMBDA_SDP = 0.8;

namespace hologainimpl {
using Eigen::MatrixXcf;
using Eigen::Vector3f;
using std::complex;

const float DIR_COEF_A[] = {1.0f, 1.0f, 1.0f, 0.891250938f, 0.707945784f, 0.501187234f, 0.354813389f, 0.251188643f, 0.199526231f};
const float DIR_COEF_B[] = {
    0.f, 0.f, -0.00459648054721f, -0.0155520765675f, -0.0208114779827f, -0.0182211227016f, -0.0122437497109f, -0.00780345575475f, -0.00312857467007f};
const float DIR_COEF_C[] = {0.f,
                            0.f,
                            -0.000787968093807f,
                            -0.000307591508224f,
                            -0.000218348633296f,
                            0.00047738416141f,
                            0.000120353137658f,
                            0.000323676257958f,
                            0.000143850511f};
const float DIR_COEF_D[] = {0.f,
                            0.f,
                            1.60125528528e-05f,
                            2.9747624976e-06f,
                            2.31910931569e-05f,
                            -1.1901034125e-05f,
                            6.77743734332e-06f,
                            -5.99548024824e-06f,
                            -4.79372835035e-06f};

static float directivity_t4010a1(float theta_deg) {
  theta_deg = abs(theta_deg);

  while (theta_deg > 90.0f) {
    theta_deg = abs(180.0f - theta_deg);
  }

  size_t i = static_cast<size_t>(ceil(theta_deg / 10.0f));

  if (i == 0) {
    return 1.0;
  } else {
    auto a = DIR_COEF_A[i - 1];
    auto b = DIR_COEF_B[i - 1];
    auto c = DIR_COEF_C[i - 1];
    auto d = DIR_COEF_D[i - 1];
    auto x = theta_deg - (i - 1.0f) * 10.0f;
    return a + b * x + c * x * x + d * x * x * x;
  }
}

complex<float> transfer(Vector3f trans_pos, Vector3f trans_norm, Vector3f target_pos) {
  const auto diff = target_pos - trans_pos;
  const auto dist = diff.norm();
  const auto theta = atan2(diff.dot(trans_norm), dist * trans_norm.norm()) * 180.0f / M_PIf;
  const auto directivity = directivity_t4010a1(theta);

  return directivity / dist * exp(complex<float>(-dist * 1.15e-4f, -2 * M_PIf / ULTRASOUND_WAVELENGTH * dist));
}

void removeRow(MatrixXcf* const matrix, size_t rowToRemove) {
  const auto numRows = static_cast<size_t>(matrix->rows()) - 1;
  const auto numCols = static_cast<size_t>(matrix->cols());

  if (rowToRemove < numRows)
    matrix->block(rowToRemove, 0, numRows - rowToRemove, numCols) = matrix->block(rowToRemove + 1, 0, numRows - rowToRemove, numCols);

  matrix->conservativeResize(numRows, numCols);
}

void removeColumn(MatrixXcf* const matrix, size_t colToRemove) {
  const auto numRows = static_cast<size_t>(matrix->rows());
  const auto numCols = static_cast<size_t>(matrix->cols()) - 1;

  if (colToRemove < numCols)
    matrix->block(0, colToRemove, numRows, numCols - colToRemove) = matrix->block(0, colToRemove + 1, numRows, numCols - colToRemove);

  matrix->conservativeResize(numRows, numCols);
}
}  // namespace hologainimpl

namespace autd {

autd::GainPtr autd::HoloGainSdp::Create(Eigen::MatrixX3f foci, Eigen::VectorXf amp) {
  auto ptr = CreateHelper<HoloGainSdp>();
  ptr->_foci = foci;
  ptr->_amp = amp;
  return ptr;
}

void autd::HoloGainSdp::build() {
  if (this->built()) return;
  auto geo = this->geometry();
  if (geo == nullptr) {
    throw std::runtime_error("Geometry is required to build Gain");
  }

  const auto alpha = 1e-3f;

  const size_t M = _foci.rows();
  const auto N = static_cast<int>(geo->numTransducers());

  Eigen::MatrixXcf P = Eigen::MatrixXcf::Zero(M, M);
  Eigen::MatrixXcf B = Eigen::MatrixXcf(M, N);

  std::random_device rnd;
  std::mt19937 mt(rnd());
  std::uniform_real_distribution<float> range(0, 1);

  for (int i = 0; i < M; i++) {
    P(i, i) = _amp(i);

    const auto tp = _foci.row(i);
    for (int j = 0; j < N; j++) {
      B(i, j) = hologainimpl::transfer(geo->position(j), geo->direction(j), tp);
    }
  }

  Eigen::JacobiSVD<Eigen::MatrixXcf> svd(B, Eigen::ComputeThinU | Eigen::ComputeThinV);
  Eigen::JacobiSVD<Eigen::MatrixXcf>::SingularValuesType singularValues_inv = svd.singularValues();
  for (int64_t i = 0; i < singularValues_inv.size(); ++i) {
    singularValues_inv(i) = singularValues_inv(i) / (singularValues_inv(i) * singularValues_inv(i) + alpha * alpha);
  }
  Eigen::MatrixXcf pinvB = (svd.matrixV() * singularValues_inv.asDiagonal() * svd.matrixU().adjoint());

  Eigen::MatrixXcf MM = P * (Eigen::MatrixXcf::Identity(M, M) - B * pinvB) * P;
  Eigen::MatrixXcf X = Eigen::MatrixXcf::Identity(M, M);
  for (int i = 0; i < M * REPEAT_SDP; i++) {
    auto ii = static_cast<size_t>(M * static_cast<double>(range(mt)));

    auto Xc = X;
    hologainimpl::removeRow(&Xc, ii);
    hologainimpl::removeColumn(&Xc, ii);
    Eigen::VectorXcf MMc = MM.col(ii);
    MMc.block(ii, 0, MMc.rows() - 1 - ii, 1) = MMc.block(ii + 1, 0, MMc.rows() - 1 - ii, 1);
    MMc.conservativeResize(MMc.rows() - 1, 1);

    Eigen::VectorXcf x = Xc * MMc;
    std::complex<float> gamma = x.adjoint() * MMc;
    if (gamma.real() > 0) {
      x = -x * sqrt(LAMBDA_SDP / gamma.real());
      X.block(ii, 0, 1, ii) = x.block(0, 0, ii, 1).adjoint().eval();
      X.block(ii, ii + 1, 1, M - ii - 1) = x.block(ii, 0, M - 1 - ii, 1).adjoint().eval();
      X.block(0, ii, ii, 1) = x.block(0, 0, ii, 1).eval();
      X.block(ii + 1, ii, M - ii - 1, 1) = x.block(ii, 0, M - 1 - ii, 1).eval();
    } else {
      X.block(ii, 0, 1, ii) = Eigen::VectorXcf::Zero(ii).adjoint();
      X.block(ii, ii + 1, 1, M - ii - 1) = Eigen::VectorXcf::Zero(M - ii - 1).adjoint();
      X.block(0, ii, ii, 1) = Eigen::VectorXcf::Zero(ii);
      X.block(ii + 1, ii, M - ii - 1, 1) = Eigen::VectorXcf::Zero(M - ii - 1);
    }
  }

  Eigen::ComplexEigenSolver<Eigen::MatrixXcf> ces(X);
  Eigen::VectorXcf evs = ces.eigenvalues();
  float abseiv = 0;
  int idx = 0;
  for (int j = 0; j < evs.rows(); j++) {
    const auto eiv = abs(evs(j));
    if (abseiv < eiv) {
      abseiv = eiv;
      idx = j;
    }
  }

  Eigen::VectorXcf u = ces.eigenvectors().col(idx);
  const auto q = pinvB * P * u;

  this->_data.clear();
  const int ndevice = geo->numDevices();
  for (int i = 0; i < ndevice; i++) {
    this->_data[geo->deviceIdForDeviceIdx(i)].resize(NUM_TRANS_IN_UNIT);
  }

  // auto maxCoeff = sqrt(q.cwiseAbs2().maxCoeff());
  for (int j = 0; j < N; j++) {
    const auto famp = 1.0f;  // abs(q(j)) / maxCoeff;
    const auto fphase = arg(q(j)) / (2 * M_PIf) + 0.5f;
    const auto amp = static_cast<uint8_t>(famp * 255);
    const auto phase = static_cast<uint8_t>((1 - fphase) * 255);
    uint8_t D, S;
    SignalDesign(amp, phase, &D, &S);
    this->_data[geo->deviceIdForTransIdx(j)].at(j % NUM_TRANS_IN_UNIT) = (static_cast<uint16_t>(D) << 8) + S;
  }
}
}  // namespace autd
