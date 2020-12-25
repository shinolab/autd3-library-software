// File: holo_gain.cpp
// Project: lib
// Created Date: 06/07/2016
// Author: Seki Inoue
// -----
// Last Modified: 25/12/2020
// Modified By: Shun Suzuki (suzuki@hapis.k.u-tokyo.ac.jp)
// -----
// Copyright (c) 2016-2020 Hapis Lab. All rights reserved.
//

#include <complex>
#include <map>
#include <random>
#include <utility>
#include <vector>

#if WIN32
#include <codeanalysis/warnings.h>  // NOLINT
#pragma warning(push)
#pragma warning(disable : ALL_CODE_ANALYSIS_WARNINGS)
#endif
#include <Eigen/Eigen>
#if WIN32
#pragma warning(pop)
#endif

#include "consts.hpp"
#include "gain.hpp"

using autd::ULTRASOUND_WAVELENGTH;

namespace hologainimpl {
using autd::GeometryPtr;
using autd::NUM_TRANS_IN_UNIT;
using autd::gain::AdjustAmp;
using Eigen::MatrixX3d, Eigen::MatrixXd, Eigen::MatrixXcd, Eigen::Vector3d, Eigen::VectorXcd, Eigen::VectorXd;
using std::complex, std::map, std::vector;

static constexpr auto ATTENUATION = 1.15e-4;

constexpr double DIR_COEFF_A[] = {1.0, 1.0, 1.0, 0.891250938, 0.707945784, 0.501187234, 0.354813389, 0.251188643, 0.199526231};
constexpr double DIR_COEFF_B[] = {
    0., 0., -0.00459648054721, -0.0155520765675, -0.0208114779827, -0.0182211227016, -0.0122437497109, -0.00780345575475, -0.00312857467007};
constexpr double DIR_COEFF_C[] = {
    0., 0., -0.000787968093807, -0.000307591508224, -0.000218348633296, 0.00047738416141, 0.000120353137658, 0.000323676257958, 0.000143850511};
constexpr double DIR_COEFF_D[] = {
    0., 0., 1.60125528528e-05, 2.9747624976e-06, 2.31910931569e-05, -1.1901034125e-05, 6.77743734332e-06, -5.99548024824e-06, -4.79372835035e-06};

static double DirectivityT4010A1(double theta_deg) {
  theta_deg = abs(theta_deg);

  while (theta_deg > 90.0) {
    theta_deg = abs(180.0 - theta_deg);
  }

  const auto i = static_cast<size_t>(ceil(theta_deg / 10.0));

  if (i == 0) {
    return 1.0;
  }

  const auto a = DIR_COEFF_A[i - 1];
  const auto b = DIR_COEFF_B[i - 1];
  const auto c = DIR_COEFF_C[i - 1];
  const auto d = DIR_COEFF_D[i - 1];
  const auto x = theta_deg - static_cast<double>(i - 1) * 10.0;
  return a + b * x + c * x * x + d * x * x * x;
}

complex<double> transfer(const Vector3d& trans_pos, const Vector3d& trans_norm, const Vector3d& target_pos) {
  const auto diff = target_pos - trans_pos;
  const auto dist = diff.norm();
  const auto theta = atan2(diff.dot(trans_norm), dist * trans_norm.norm()) * 180.0 / M_PI;
  const auto directivity = DirectivityT4010A1(theta);

  return directivity / dist * exp(complex<double>(-dist * ATTENUATION, -2 * M_PI / ULTRASOUND_WAVELENGTH * dist));
}

void removeRow(MatrixXcd* const matrix, const size_t row_to_remove) {
  const auto num_rows = static_cast<size_t>(matrix->rows()) - 1;
  const auto num_cols = static_cast<size_t>(matrix->cols());

  if (row_to_remove < num_rows)
    matrix->block(row_to_remove, 0, num_rows - row_to_remove, num_cols) = matrix->block(row_to_remove + 1, 0, num_rows - row_to_remove, num_cols);

  matrix->conservativeResize(num_rows, num_cols);
}

void removeColumn(MatrixXcd* const matrix, const size_t col_to_remove) {
  const auto num_rows = static_cast<size_t>(matrix->rows());
  const auto num_cols = static_cast<size_t>(matrix->cols()) - 1;

  if (col_to_remove < num_cols)
    matrix->block(0, col_to_remove, num_rows, num_cols - col_to_remove) = matrix->block(0, col_to_remove + 1, num_rows, num_cols - col_to_remove);

  matrix->conservativeResize(num_rows, num_cols);
}

MatrixXcd TransferMatrix(const GeometryPtr& geometry, const MatrixX3d& foci, const size_t M, const size_t N) {
  auto G = MatrixXcd(M, N);

  for (size_t i = 0; i < M; i++) {
    const auto tp = foci.row(i);
    for (size_t j = 0; j < N; j++) {
      const auto pos = geometry->position(j);
      const auto dir = geometry->direction(j);
      G(i, j) = transfer(Vector3d(pos.x(), pos.y(), pos.z()), Vector3d(dir.x(), dir.y(), dir.z()), tp);
    }
  }

  return G;
}

void HoloGainImplSDP(vector<vector<uint16_t>>* data, const MatrixX3d& foci, const VectorXd& amps, const autd::GeometryPtr& geometry, void* params) {
  double alpha, lambda;
  int32_t repeat;
  bool normalize;

  if (params != nullptr) {
    const auto sdp_params = static_cast<autd::gain::SDPParams*>(params);
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
  const auto N = geometry->numTransducers();

  MatrixXcd P = MatrixXcd::Zero(M, M);
  for (size_t i = 0; i < M; i++) {
    P(i, i) = amps(i);
  }

  const auto B = TransferMatrix(geometry, foci, M, N);

  std::random_device seed_gen;
  std::mt19937 mt(seed_gen());
  std::uniform_real_distribution<double> range(0, 1);

  const Eigen::JacobiSVD<MatrixXcd> svd(B, Eigen::ComputeThinU | Eigen::ComputeThinV);
  auto singularValues_inv = svd.singularValues();
  for (int64_t i = 0; i < singularValues_inv.size(); ++i) {
    singularValues_inv(i) = singularValues_inv(i) / (singularValues_inv(i) * singularValues_inv(i) + alpha * alpha);
  }
  const MatrixXcd p_inv_B = (svd.matrixV() * singularValues_inv.asDiagonal() * svd.matrixU().adjoint());

  MatrixXcd MM = P * (MatrixXcd::Identity(M, M) - B * p_inv_B) * P;
  MatrixXcd X = MatrixXcd::Identity(M, M);
  for (auto i = 0; i < repeat; i++) {
    const double rnd = range(mt);
    const auto ii = static_cast<size_t>(static_cast<double>(M) * rnd);

    auto Xc = X;
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

  const Eigen::ComplexEigenSolver<MatrixXcd> ces(X);
  auto evs = ces.eigenvalues();
  double abs_eiv = 0;
  auto idx = 0;
  for (auto j = 0; j < evs.rows(); j++) {
    const auto eiv = abs(evs(j));
    if (abs_eiv < eiv) {
      abs_eiv = eiv;
      idx = j;
    }
  }

  const VectorXcd u = ces.eigenvectors().col(idx);
  const auto q = p_inv_B * P * u;

  const auto max_coeff = sqrt(q.cwiseAbs2().maxCoeff());
  for (size_t j = 0; j < N; j++) {
    const auto f_amp = normalize ? 1.0 : abs(q(j)) / max_coeff;
    const auto f_phase = arg(q(j)) / (2 * M_PI) + 0.5;
    const auto phase = static_cast<uint16_t>((1 - f_phase) * 255.);
    const auto D = (static_cast<uint16_t>(AdjustAmp(f_amp)) << 8) & 0xFF00;
    data->at(geometry->deviceIdxForTransIdx(j)).at(j % NUM_TRANS_IN_UNIT) = D | phase;
  }
}

void HoloGainImplEVD(vector<vector<uint16_t>>* data, const MatrixX3d& foci, const VectorXd& amps, const GeometryPtr& geometry, void* params) {
  double gamma;
  bool normalize;

  if (params != nullptr) {
    auto evd_params = static_cast<autd::gain::EVDParams*>(params);
    gamma = evd_params->regularization < 0 ? 1.0 : evd_params->regularization;
    normalize = evd_params->normalize_amp;
  } else {
    gamma = 1;
    normalize = true;
  }

  const size_t M = foci.rows();
  const auto N = geometry->numTransducers();

  auto G = TransferMatrix(geometry, foci, M, N);

  VectorXcd denominator(M);
  for (size_t i = 0; i < M; i++) {
    auto tmp = complex<double>(0, 0);
    for (size_t j = 0; j < N; j++) {
      tmp += G(i, j);
    }
    denominator(i) = tmp;
  }

  MatrixXcd X(N, M);
  for (size_t i = 0; i < M; i++) {
    auto c = complex<double>(amps(i), 0) / denominator(i);
    for (size_t j = 0; j < N; j++) {
      X(j, i) = c * std::conj(G(i, j));
    }
  }
  auto R = G * X;

  Eigen::ComplexEigenSolver<MatrixXcd> ces(R);
  auto evs = ces.eigenvalues();
  double abs_eiv = 0;
  auto idx = 0;
  for (auto j = 0; j < evs.rows(); j++) {
    const auto eiv = abs(evs(j));
    if (abs_eiv < eiv) {
      abs_eiv = eiv;
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
    sigma(j, j) = complex<double>(pow(sqrt(tmp / static_cast<double>(M)), gamma), 0.0);
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

  auto max_coeff = sqrt(q.cwiseAbs2().maxCoeff());
  for (size_t j = 0; j < N; j++) {
    const auto f_amp = normalize ? 1.0 : abs(q(j)) / max_coeff;
    const auto f_phase = arg(q(j)) / (2 * M_PI) + 0.5;
    const auto phase = static_cast<uint16_t>((1 - f_phase) * 255.);
    const auto D = (static_cast<uint16_t>(AdjustAmp(f_amp)) << 8) & 0xFF00;
    data->at(geometry->deviceIdxForTransIdx(j)).at(j % NUM_TRANS_IN_UNIT) = D | phase;
  }
}

void HoloGainImplNaive(vector<vector<uint16_t>>* data, const MatrixX3d& foci, const VectorXd& amps, const GeometryPtr& geometry, void* params) {
  const size_t M = foci.rows();
  const auto N = geometry->numTransducers();

  const auto G = TransferMatrix(geometry, foci, M, N);
  const auto Gh = G.adjoint();
  const VectorXcd p = amps;
  const auto q = Gh * p;
  for (size_t j = 0; j < N; j++) {
    const auto f_amp = abs(q(j));
    const auto f_phase = arg(q(j)) / (2 * M_PI) + 0.5;
    const auto phase = static_cast<uint8_t>((1 - f_phase) * 255.);
    const auto D = (static_cast<uint16_t>(AdjustAmp(f_amp)) << 8) & 0xFF00;
    data->at(geometry->deviceIdxForTransIdx(j)).at(j % NUM_TRANS_IN_UNIT) = D | phase;
  }
}

void HoloGainImplGS(vector<vector<uint16_t>>* data, const MatrixX3d& foci, const VectorXd& amps, const GeometryPtr& geometry, void* params) {
  const int32_t repeat = (params == nullptr) ? 100 : *static_cast<uint32_t*>(params);

  const size_t M = foci.rows();
  const auto N = geometry->numTransducers();

  const auto G = TransferMatrix(geometry, foci, M, N);

  const auto Gh = G.adjoint();

  VectorXcd p0 = amps;
  VectorXcd q0 = VectorXcd::Ones(N);

  auto q = q0;
  for (auto k = 0; k < repeat; k++) {
    auto gamma = G * q;
    VectorXcd p(M);
    for (size_t i = 0; i < M; i++) p(i) = gamma(i) / abs(gamma(i)) * p0(i);
    auto xi = Gh * p;
    for (size_t j = 0; j < N; j++) q(j) = xi(j) / abs(xi(j)) * q0(j);
  }

  for (auto j = 0; j < N; j++) {
    const auto f_amp = abs(q(j));
    const auto f_phase = arg(q(j)) / (2 * M_PI) + 0.5;
    const auto phase = static_cast<uint8_t>((1 - f_phase) * 255.);
    const auto D = (static_cast<uint16_t>(AdjustAmp(f_amp)) << 8) & 0xFF00;
    data->at(geometry->deviceIdxForTransIdx(j)).at(j % NUM_TRANS_IN_UNIT) = D | phase;
  }
}

void HoloGainImplGSPAT(vector<vector<uint16_t>>* data, const MatrixX3d& foci, const VectorXd& amps, const GeometryPtr& geometry, void* params) {
  const int32_t repeat = (params == nullptr) ? 100 : *static_cast<uint32_t*>(params);

  const size_t M = foci.rows();
  const auto N = geometry->numTransducers();

  auto G = TransferMatrix(geometry, foci, M, N);

  VectorXcd denominator(M);
  for (size_t i = 0; i < M; i++) {
    auto tmp = complex<double>(0, 0);
    for (size_t j = 0; j < N; j++) tmp += abs(G(i, j));
    denominator(i) = tmp;
  }

  MatrixXcd B(N, M);
  for (size_t i = 0; i < M; i++) {
    auto d = (denominator(i) * denominator(i));
    for (size_t j = 0; j < N; j++) {
      B(j, i) = std::conj(G(i, j)) / d;
    }
  }

  const auto R = G * B;

  VectorXcd p0 = amps;
  auto p = p0;
  VectorXcd gamma = R * p;
  for (auto k = 0; k < repeat; k++) {
    for (size_t i = 0; i < M; i++) p(i) = gamma(i) / abs(gamma(i)) * p0(i);
    gamma = R * p;
  }

  for (size_t i = 0; i < M; i++) p(i) = gamma(i) / (abs(gamma(i)) * abs(gamma(i))) * p0(i) * p0(i);
  const auto q = B * p;

  for (size_t j = 0; j < N; j++) {
    const auto f_amp = abs(q(j));
    const auto f_phase = arg(q(j)) / (2 * M_PI) + 0.5;
    const auto phase = static_cast<uint16_t>((1 - f_phase) * 255.);
    const auto duty = (static_cast<uint16_t>(AdjustAmp(f_amp)) << 8) & 0xFF00;
    data->at(geometry->deviceIdxForTransIdx(j)).at(j % NUM_TRANS_IN_UNIT) = duty | phase;
  }
}

void HoloGainImplLM(vector<vector<uint16_t>>* data, const MatrixX3d& foci, const VectorXd& amps, const GeometryPtr& geometry, void* params) {
  double eps_1, eps_2, tau;
  int32_t k_max;

  if (params != nullptr) {
    auto nlp_params = static_cast<autd::gain::NLSParams*>(params);
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
  const auto N = geometry->numTransducers();
  const auto n_param = N + M;

  MatrixXcd P = MatrixXcd::Zero(M, M);
  for (size_t i = 0; i < M; i++) P(i, i) = -amps(i);

  auto G = TransferMatrix(geometry, foci, M, N);

  VectorXd x0 = VectorXd::Zero(n_param);
  MatrixXd I = MatrixXd::Identity(n_param, n_param);

  MatrixXcd B(G.rows(), G.cols() + P.cols());
  B << G, P;
  auto BhB = B.adjoint() * B;

  auto x = x0;
  auto nu = 2.0;

  VectorXcd T(n_param);
  for (size_t i = 0; i < n_param; i++) T(i) = exp(complex<double>(0, -x(i)));

  MatrixXcd TTh = T * T.adjoint();
  MatrixXcd BhB_TTh = BhB.cwiseProduct(TTh);
  MatrixXd A = BhB_TTh.real();
  VectorXd g(n_param);
  for (size_t i = 0; i < n_param; i++) {
    auto tmp = 0.0;
    for (size_t k = 0; k < n_param; k++) tmp += BhB_TTh(i, k).imag();
    g(i) = tmp;
  }

  auto a_max = A.diagonal().maxCoeff();
  auto mu = tau * a_max;

  auto is_found = (g.maxCoeff() <= eps_1);

  VectorXcd t(n_param);
  for (size_t i = 0; i < n_param; i++) t(i) = exp(complex<double>(0, x(i)));
  auto fx = (t.adjoint() * BhB * t)[0].real();

  for (auto k = 0; k < k_max; k++) {
    if (is_found) break;

    Eigen::HouseholderQR<MatrixXd> qr(A + mu * I);
    auto h_lm = -qr.solve(g);
    if (h_lm.norm() <= eps_2 * (x.norm() + eps_2)) {
      is_found = true;
    } else {
      auto x_new = x + h_lm;
      for (size_t i = 0; i < n_param; i++) t(i) = exp(complex<double>(0, x_new(i)));
      auto Fx_new = (t.adjoint() * BhB * t)[0].real();
      auto l0_lhlm = 0.5 * h_lm.dot(mu * h_lm - g);
      auto rho = (fx - Fx_new) / l0_lhlm;
      fx = Fx_new;
      if (rho > 0.0) {
        x = x_new;
        for (size_t i = 0; i < n_param; i++) T(i) = exp(complex<double>(0, -x(i)));
        TTh = T * T.adjoint();
        BhB_TTh = BhB.cwiseProduct(TTh);
        A = BhB_TTh.real();
        for (size_t i = 0; i < n_param; i++) {
          auto tmp = 0.0;
          for (size_t j = 0; j < n_param; j++) tmp += BhB_TTh(i, j).imag();
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

  const uint16_t D = 0xFF00;
  for (size_t j = 0; j < N; j++) {
    const auto f_phase = fmod(x(j), 2 * M_PI) / (2 * M_PI);
    const auto S = static_cast<uint16_t>((1 - f_phase) * 255.);
    data->at(geometry->deviceIdxForTransIdx(j)).at(j % NUM_TRANS_IN_UNIT) = D | S;
  }
}
}  // namespace hologainimpl

namespace autd::gain {

GainPtr HoloGain::Create(const std::vector<Vector3>& foci, const std::vector<double>& amps, const OptMethod method, void* params) {
  GainPtr ptr = std::make_shared<HoloGain>(foci, amps, method, params);
  return ptr;
}

void HoloGain::Build() {
  if (this->built()) return;
  const auto geo = this->geometry();

  CheckAndInit(geo, &this->_data);

  const auto M = _foci.size();

  Eigen::MatrixX3d foci(M, 3);
  Eigen::VectorXd amps(M);

  for (size_t i = 0; i < M; i++) {
    foci(i, 0) = _foci[i].x();
    foci(i, 1) = _foci[i].y();
    foci(i, 2) = _foci[i].z();
    amps(i) = _amps[i];
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
