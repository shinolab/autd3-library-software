// File: holo_gain.cpp
// Project: lib
// Created Date: 06/07/2016
// Author: Seki Inoue
// -----
// Last Modified: 10/12/2020
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

static constexpr auto ATTENUATION = 1.15e-4;

namespace hologainimpl {
using autd::GeometryPtr;
using autd::NUM_TRANS_IN_UNIT;
using autd::gain::AdjustAmp;
using Eigen::MatrixX3d, Eigen::MatrixXd, Eigen::MatrixXcd, Eigen::Vector3d, Eigen::VectorXcd, Eigen::VectorXd;
using std::complex, std::map, std::vector;

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

complex<double> transfer(Vector3d trans_pos, Vector3d trans_norm, Vector3d target_pos) {
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

MatrixXcd TrnasferMatrix(GeometryPtr geometry, const MatrixX3d& foci, size_t M, size_t N) {
  MatrixXcd G = MatrixXcd(M, N);

  for (size_t i = 0; i < M; i++) {
    const auto tp = foci.row(i);
    for (int j = 0; j < N; j++) {
      const auto pos = geometry->position(j);
      const auto dir = geometry->direction(j);
      G(i, j) = transfer(Vector3d(pos.x(), pos.y(), pos.z()), Vector3d(dir.x(), dir.y(), dir.z()), tp);
    }
  }

  return G;
}

void HoloGainImplSDP(vector<vector<uint16_t>>* data, const MatrixX3d& foci, const VectorXd& amps, autd::GeometryPtr geometry, void* params) {
  double alpha, lambda;
  int32_t repeat;
  bool normalize;

  if (params != nullptr) {
    auto sdp_params = reinterpret_cast<autd::gain::SDPParams*>(params);
    alpha = sdp_params->regularization < 0 ? 1e-3 : sdp_params->regularization;
    repeat = sdp_params->repeat < 0 ? 10 : sdp_params->repeat;
    lambda = sdp_params->lambda < 0 ? 0.8 : sdp_params->lambda;
    normalize = sdp_params->normalize_amp;
  } else {
    alpha = 1e-3;
    repeat = 10;
    lambda = 0.8;
    normalize = true;
  }

  const size_t M = foci.rows();
  const auto N = static_cast<int>(geometry->numTransducers());

  MatrixXcd P = MatrixXcd::Zero(M, M);
  for (size_t i = 0; i < M; i++) {
    P(i, i) = amps(i);
  }

  MatrixXcd B = TrnasferMatrix(geometry, foci, M, N);

  std::random_device rnd;
  std::mt19937 mt(rnd());
  std::uniform_real_distribution<double> range(0, 1);

  Eigen::JacobiSVD<MatrixXcd> svd(B, Eigen::ComputeThinU | Eigen::ComputeThinV);
  Eigen::JacobiSVD<MatrixXcd>::SingularValuesType singularValues_inv = svd.singularValues();
  for (int64_t i = 0; i < singularValues_inv.size(); ++i) {
    singularValues_inv(i) = singularValues_inv(i) / (singularValues_inv(i) * singularValues_inv(i) + alpha * alpha);
  }
  MatrixXcd pinvB = (svd.matrixV() * singularValues_inv.asDiagonal() * svd.matrixU().adjoint());

  MatrixXcd MM = P * (MatrixXcd::Identity(M, M) - B * pinvB) * P;
  MatrixXcd X = MatrixXcd::Identity(M, M);
  for (int32_t i = 0; i < repeat; i++) {
    auto ii = static_cast<size_t>(M * static_cast<double>(range(mt)));

    MatrixXcd Xc = X;
    removeRow(&Xc, ii);
    removeColumn(&Xc, ii);
    VectorXcd MMc = MM.col(ii);
    MMc.block(ii, 0, MMc.rows() - 1 - ii, 1) = MMc.block(ii + 1, 0, MMc.rows() - 1 - ii, 1);
    MMc.conservativeResize(MMc.rows() - 1, 1);

    VectorXcd x = Xc * MMc;
    complex<double> gamma = x.adjoint() * MMc;
    if (gamma.real() > 0) {
      x = -x * sqrt(lambda / gamma.real());
      X.block(ii, 0, 1, ii) = x.block(0, 0, ii, 1).adjoint().eval();
      X.block(ii, ii + 1, 1, M - ii - 1) = x.block(ii, 0, M - 1 - ii, 1).adjoint().eval();
      X.block(0, ii, ii, 1) = x.block(0, 0, ii, 1).eval();
      X.block(ii + 1, ii, M - ii - 1, 1) = x.block(ii, 0, M - 1 - ii, 1).eval();
    } else {
      X.block(ii, 0, 1, ii) = VectorXcd::Zero(ii).adjoint();
      X.block(ii, ii + 1, 1, M - ii - 1) = VectorXcd::Zero(M - ii - 1).adjoint();
      X.block(0, ii, ii, 1) = VectorXcd::Zero(ii);
      X.block(ii + 1, ii, M - ii - 1, 1) = VectorXcd::Zero(M - ii - 1);
    }
  }

  Eigen::ComplexEigenSolver<MatrixXcd> ces(X);
  VectorXcd evs = ces.eigenvalues();
  double abseiv = 0;
  int idx = 0;
  for (int j = 0; j < evs.rows(); j++) {
    const auto eiv = abs(evs(j));
    if (abseiv < eiv) {
      abseiv = eiv;
      idx = j;
    }
  }

  VectorXcd u = ces.eigenvectors().col(idx);
  const auto q = pinvB * P * u;

  auto maxCoeff = sqrt(q.cwiseAbs2().maxCoeff());
  for (int j = 0; j < N; j++) {
    const auto famp = normalize ? 1.0 : abs(q(j)) / maxCoeff;
    const auto fphase = arg(q(j)) / (2 * M_PI) + 0.5;
    const auto phase = static_cast<uint8_t>((1 - fphase) * 255);
    uint8_t S = phase;
    uint8_t D = AdjustAmp(famp);
    data->at(geometry->deviceIdxForTransIdx(j)).at(j % NUM_TRANS_IN_UNIT) = (static_cast<uint16_t>(D) << 8) + S;
  }
}

void HoloGainImplEVD(vector<vector<uint16_t>>* data, const MatrixX3d& foci, const VectorXd& amps, GeometryPtr geometry, void* params) {
  double gamma;
  bool normalize;

  if (params != nullptr) {
    auto evd_params = reinterpret_cast<autd::gain::EVDParams*>(params);
    gamma = evd_params->regularization < 0 ? 1.0 : evd_params->regularization;
    normalize = evd_params->normalize_amp;
  } else {
    gamma = 1;
    normalize = true;
  }

  const size_t M = foci.rows();
  const auto N = static_cast<int>(geometry->numTransducers());

  auto G = TrnasferMatrix(geometry, foci, M, N);

  VectorXcd denomi(M);
  for (size_t i = 0; i < M; i++) {
    auto tmp = complex<double>(0, 0);
    for (size_t j = 0; j < N; j++) {
      tmp += G(i, j);
    }
    denomi(i) = tmp;
  }

  MatrixXcd X(N, M);
  for (size_t i = 0; i < M; i++) {
    auto c = complex<double>(amps(i), 0) / denomi(i);
    for (size_t j = 0; j < N; j++) {
      X(j, i) = c * std::conj(G(i, j));
    }
  }
  auto R = G * X;

  Eigen::ComplexEigenSolver<MatrixXcd> ces(R);
  VectorXcd evs = ces.eigenvalues();
  double abseiv = 0;
  int idx = 0;
  for (int j = 0; j < evs.rows(); j++) {
    const auto eiv = abs(evs(j));
    if (abseiv < eiv) {
      abseiv = eiv;
      idx = j;
    }
  }
  auto max_ev = ces.eigenvectors().row(idx);
  VectorXd e_arg(M);
  for (size_t i = 0; i < M; i++) {
    e_arg(i) = arg(max_ev(i));
  }

  auto sigma = MatrixXcd(N, N);
  for (size_t j = 0; j < N; j++) {
    auto tmp = 0.0;
    for (size_t i = 0; i < M; i++) {
      tmp += abs(G(i, j)) * amps(i);
    }
    sigma(j, j) = complex<double>(pow(sqrt(tmp / M), gamma), 0.0);
  }

  MatrixXcd g(G.rows() + sigma.rows(), G.cols());
  g << G, sigma;

  VectorXcd f = VectorXcd::Zero(M + N);
  for (size_t i = 0; i < M; i++) {
    f(i) = amps(i) * exp(complex<double>(0.0, e_arg(i)));
  }

  auto gt = g.adjoint();
  auto gtg = gt * g;
  auto gtf = gt * f;
  Eigen::FullPivHouseholderQR<MatrixXcd> qr(gtg);
  auto q = qr.solve(gtf);

  auto maxCoeff = sqrt(q.cwiseAbs2().maxCoeff());
  for (int j = 0; j < N; j++) {
    const auto famp = normalize ? 1.0 : abs(q(j)) / maxCoeff;
    const auto fphase = arg(q(j)) / (2 * M_PI) + 0.5;
    const auto phase = static_cast<uint8_t>((1 - fphase) * 255);
    uint8_t S = phase;
    uint8_t D = AdjustAmp(famp);
    data->at(geometry->deviceIdxForTransIdx(j)).at(j % NUM_TRANS_IN_UNIT) = (static_cast<uint16_t>(D) << 8) + S;
  }
}

void HoloGainImplNaive(vector<vector<uint16_t>>* data, const MatrixX3d& foci, const VectorXd& amps, GeometryPtr geometry, void* params) {
  const size_t M = foci.rows();
  const auto N = static_cast<int>(geometry->numTransducers());

  auto G = TrnasferMatrix(geometry, foci, M, N);
  auto Gh = G.adjoint();
  VectorXcd p = amps;
  auto q = Gh * p;
  for (int j = 0; j < N; j++) {
    const auto famp = abs(q(j));
    const auto fphase = arg(q(j)) / (2 * M_PI) + 0.5;
    const auto phase = static_cast<uint8_t>((1 - fphase) * 255);
    uint8_t S = phase;
    uint8_t D = AdjustAmp(famp);
    data->at(geometry->deviceIdxForTransIdx(j)).at(j % NUM_TRANS_IN_UNIT) = (static_cast<uint16_t>(D) << 8) + S;
  }
}

void HoloGainImplGS(vector< vector<uint16_t>>* data, const MatrixX3d& foci, const VectorXd& amps, GeometryPtr geometry, void* params) {
  const int32_t repeat = (params == nullptr) ? 100 : *reinterpret_cast<uint32_t*>(params);

  const size_t M = foci.rows();
  const auto N = static_cast<int>(geometry->numTransducers());

  auto G = TrnasferMatrix(geometry, foci, M, N);

  auto Gh = G.adjoint();

  VectorXcd p0 = amps;
  VectorXcd q0 = VectorXcd::Ones(N);

  VectorXcd q = q0;
  for (int32_t k = 0; k < repeat; k++) {
    auto gamma = G * q;
    VectorXcd p(M);
    for (size_t i = 0; i < M; i++) p(i) = gamma(i) / abs(gamma(i)) * p0(i);
    auto xi = Gh * p;
    for (size_t j = 0; j < N; j++) q(j) = xi(j) / abs(xi(j)) * q0(j);
  }

  for (int j = 0; j < N; j++) {
    const auto famp = abs(q(j));
    const auto fphase = arg(q(j)) / (2 * M_PI) + 0.5;
    const auto phase = static_cast<uint8_t>((1 - fphase) * 255);
    uint8_t S = phase;
    uint8_t D = AdjustAmp(famp);
    data->at(geometry->deviceIdxForTransIdx(j)).at(j % NUM_TRANS_IN_UNIT) = (static_cast<uint16_t>(D) << 8) + S;
  }
}

void HoloGainImplGSPAT(vector<vector<uint16_t>>* data, const MatrixX3d& foci, const VectorXd& amps, GeometryPtr geometry, void* params) {
  const int32_t repeat = (params == nullptr) ? 100 : *reinterpret_cast<uint32_t*>(params);

  const size_t M = foci.rows();
  const auto N = static_cast<int>(geometry->numTransducers());

  auto G = TrnasferMatrix(geometry, foci, M, N);

  VectorXcd denomi(M);
  for (size_t i = 0; i < M; i++) {
    auto tmp = complex<double>(0, 0);
    for (size_t j = 0; j < N; j++) tmp += abs(G(i, j));
    denomi(i) = tmp;
  }

  MatrixXcd B(N, M);
  for (size_t i = 0; i < M; i++) {
    auto d = (denomi(i) * denomi(i));
    for (size_t j = 0; j < N; j++) {
      B(j, i) = std::conj(G(i, j)) / d;
    }
  }

  auto R = G * B;

  VectorXcd p0 = amps;
  VectorXcd p = p0;
  VectorXcd gamma = R * p;
  for (int32_t k = 0; k < repeat; k++) {
    for (size_t i = 0; i < M; i++) p(i) = gamma(i) / abs(gamma(i)) * p0(i);
    gamma = R * p;
  }

  for (size_t i = 0; i < M; i++) p(i) = gamma(i) / (abs(gamma(i)) * abs(gamma(i))) * p0(i) * p0(i);
  auto q = B * p;

  for (int j = 0; j < N; j++) {
    const auto famp = abs(q(j));
    const auto fphase = arg(q(j)) / (2 * M_PI) + 0.5;
    const auto phase = static_cast<uint8_t>((1 - fphase) * 255);
    uint8_t S = phase;
    uint8_t D = AdjustAmp(famp);
    data->at(geometry->deviceIdxForTransIdx(j)).at(j % NUM_TRANS_IN_UNIT) = (static_cast<uint16_t>(D) << 8) + S;
  }
}

void HoloGainImplLM(vector< vector<uint16_t>>* data, const MatrixX3d& foci, const VectorXd& amps, GeometryPtr geometry, void* params) {
  double eps_1, eps_2, tau;
  int32_t k_max;

  if (params != nullptr) {
    auto nlp_params = reinterpret_cast<autd::gain::NLSParams*>(params);
    eps_1 = nlp_params->eps_1 < 0 ? 1e-8 : nlp_params->eps_1;
    eps_2 = nlp_params->eps_2 < 0 ? 1e-8 : nlp_params->eps_2;
    k_max = nlp_params->k_max < 0 ? 5 : nlp_params->k_max;
    tau = nlp_params->tau < 0 ? 1e-3 : nlp_params->tau;
  } else {
    eps_1 = 1e-8;
    eps_2 = 1e-8;
    k_max = 5;
    tau = 1e-3;
  }

  const size_t M = foci.rows();
  const auto N = static_cast<int>(geometry->numTransducers());
  const auto n_param = N + M;

  MatrixXcd P = MatrixXcd::Zero(M, M);
  for (size_t i = 0; i < M; i++) P(i, i) = -amps(i);

  auto G = TrnasferMatrix(geometry, foci, M, N);

  VectorXd x0 = VectorXd::Zero(n_param);
  MatrixXd I = MatrixXd::Identity(n_param, n_param);

  MatrixXcd B(G.rows(), G.cols() + P.cols());
  B << G, P;
  auto BhB = B.adjoint() * B;

  VectorXd x = x0;
  auto nu = 2.0;

  VectorXcd T(n_param);
  for (size_t i = 0; i < n_param; i++) T(i) = exp(complex<double>(0, -x(i)));

  MatrixXcd TTh = T * T.adjoint();
  MatrixXcd BhB_TTh = BhB.cwiseProduct(TTh);
  MatrixXd A = BhB_TTh.real();
  VectorXd g(n_param);
  for (size_t i = 0; i < n_param; i++) {
    double tmp = 0.0;
    for (size_t k = 0; k < n_param; k++) tmp += BhB_TTh(i, k).imag();
    g(i) = tmp;
  }

  auto A_max = A.diagonal().maxCoeff();
  auto mu = tau * A_max;

  auto is_found = (g.maxCoeff() <= eps_1);

  VectorXcd t(n_param);
  for (size_t i = 0; i < n_param; i++) t(i) = exp(complex<double>(0, x(i)));
  double Fx = (t.adjoint() * BhB * t)[0].real();

  for (int32_t k = 0; k < k_max; k++) {
    if (is_found) break;

    Eigen::HouseholderQR<MatrixXd> qr(A + mu * I);
    auto h_lm = -qr.solve(g);
    if (h_lm.norm() <= eps_2 * (x.norm() + eps_2)) {
      is_found = true;
    } else {
      auto x_new = x + h_lm;
      for (size_t i = 0; i < n_param; i++) t(i) = exp(complex<double>(0, x_new(i)));
      double Fx_new = (t.adjoint() * BhB * t)[0].real();
      auto L0_Lhlm = 0.5 * h_lm.dot(mu * h_lm - g);
      auto rho = (Fx - Fx_new) / L0_Lhlm;
      Fx = Fx_new;
      if (rho > 0.0) {
        x = x_new;
        for (size_t i = 0; i < n_param; i++) T(i) = exp(complex<double>(0, -x(i)));
        TTh = T * T.adjoint();
        BhB_TTh = BhB.cwiseProduct(TTh);
        A = BhB_TTh.real();
        for (size_t i = 0; i < n_param; i++) {
          double tmp = 0.0;
          for (size_t k = 0; k < n_param; k++) tmp += BhB_TTh(i, k).imag();
          g(i) = tmp;
        }
        is_found = (g.maxCoeff() <= eps_1);
        mu *= std::max(1.0 / 3., pow(1. - (2.0 * rho - 1.), 3));
        nu = 2.0;
      } else {
        mu *= nu;
        nu *= 2.0;
      }
    }
  }

  const auto duty = 0xFF;
  for (int j = 0; j < N; j++) {
    const auto fphase = fmod(x(j), 2 * M_PI) / (2 * M_PI);
    const auto S = static_cast<uint8_t>((1 - fphase) * 255);
    data->at(geometry->deviceIdxForTransIdx(j)).at(j % NUM_TRANS_IN_UNIT) = (static_cast<uint16_t>(duty) << 8) + S;
  }
}
}  // namespace hologainimpl

namespace autd::gain {

GainPtr HoloGain::Create(std::vector<Vector3> foci, std::vector<double> amps, OptMethod method, void* params) {
  auto ptr = std::make_shared<HoloGain>();
  ptr->_foci = foci;
  ptr->_amps = amps;
  ptr->_method = method;
  ptr->_params = params;
  return ptr;
}

void HoloGain::Build() {
  if (this->built()) return;
  auto geo = this->geometry();
  if (geo == nullptr) {
    throw std::runtime_error("Geometry is required to build Gain");
  }

  auto M = _foci.size();

  Eigen::MatrixX3d foci(M, 3);
  Eigen::VectorXd amps(M);

  for (size_t i = 0; i < M; i++) {
    foci(i, 0) = _foci[i].x();
    foci(i, 1) = _foci[i].y();
    foci(i, 2) = _foci[i].z();
    amps(i) = _amps[i];
  }

  _data.clear();
  const int ndevice = geo->numDevices();
  for (int i = 0; i < ndevice; i++) {
    _data[i].resize(NUM_TRANS_IN_UNIT);
  }

  switch (this->_method) {
    case OptMethod::SDP:
      hologainimpl::HoloGainImplSDP(&_data, foci, amps, geo, _params);
      break;
    case OptMethod::EVD:
      hologainimpl::HoloGainImplEVD(&_data, foci, amps, geo, _params);
      break;
    case OptMethod::NAIVE:
      hologainimpl::HoloGainImplNaive(&_data, foci, amps, geo, _params);
      break;
    case OptMethod::GS:
      hologainimpl::HoloGainImplGS(&_data, foci, amps, geo, _params);
      break;
    case OptMethod::GS_PAT:
      hologainimpl::HoloGainImplGSPAT(&_data, foci, amps, geo, _params);
      break;
    case OptMethod::LM:
      hologainimpl::HoloGainImplLM(&_data, foci, amps, geo, _params);
      break;
  }
}
}  // namespace autd::gain
