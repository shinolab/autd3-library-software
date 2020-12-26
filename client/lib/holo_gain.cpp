﻿// File: holo_gain.cpp
// Project: lib
// Created Date: 06/07/2016
// Author: Seki Inoue
// -----
// Last Modified: 26/12/2020
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
#include "convert.hpp"
#include "gain.hpp"

using autd::ULTRASOUND_WAVELENGTH;

namespace hologainimpl {
using autd::AUTDDataArray;
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

complex<double> Transfer(const autd::Vector3& trans_pos, const autd::Vector3& trans_norm, const autd::Vector3& target_pos) {
  const auto diff = target_pos - trans_pos;
  const auto dist = diff.norm();
  const auto theta = atan2(diff.dot(trans_norm), dist * trans_norm.norm()) * 180.0 / M_PI;
  const auto directivity = DirectivityT4010A1(theta);

  return directivity / dist * exp(complex<double>(-dist * ATTENUATION, -2 * M_PI / ULTRASOUND_WAVELENGTH * dist));
}

void RemoveRow(MatrixXcd* const matrix, const size_t row_to_remove) {
  const auto num_rows = static_cast<size_t>(matrix->rows()) - 1;
  const auto num_cols = static_cast<size_t>(matrix->cols());

  if (row_to_remove < num_rows)
    matrix->block(row_to_remove, 0, num_rows - row_to_remove, num_cols) = matrix->block(row_to_remove + 1, 0, num_rows - row_to_remove, num_cols);

  matrix->conservativeResize(num_rows, num_cols);
}

void RemoveColumn(MatrixXcd* const matrix, const size_t col_to_remove) {
  const auto num_rows = static_cast<size_t>(matrix->rows());
  const auto num_cols = static_cast<size_t>(matrix->cols()) - 1;

  if (col_to_remove < num_cols)
    matrix->block(0, col_to_remove, num_rows, num_cols - col_to_remove) = matrix->block(0, col_to_remove + 1, num_rows, num_cols - col_to_remove);

  matrix->conservativeResize(num_rows, num_cols);
}

MatrixXcd TransferMatrix(const GeometryPtr& geometry, const MatrixX3d& foci, const size_t m, const size_t n) {
  auto g = MatrixXcd(m, n);

  for (size_t i = 0; i < m; i++) {
    const auto tp = foci.row(i);
    for (size_t j = 0; j < n; j++) {
      const auto pos = geometry->position(j);
      const auto dir = geometry->direction(j / NUM_TRANS_IN_UNIT);
      g(i, j) = Transfer(autd::Convert(pos), autd::Convert(dir), autd::Convert(tp));
    }
  }

  return g;
}

void SetFromComplexDrive(vector<AUTDDataArray>& data, const VectorXcd& drive, const bool normalize, const double max_coeff) {
  const size_t n = drive.size();
  size_t dev_idx = 0;
  size_t trans_idx = 0;
  for (size_t j = 0; j < n; j++) {
    const auto f_amp = normalize ? 1.0 : abs(drive(j)) / max_coeff;
    const auto f_phase = arg(drive(j)) / (2 * M_PI) + 0.5;
    const auto phase = static_cast<uint16_t>((1 - f_phase) * 255.);
    const uint16_t duty = static_cast<uint16_t>(AdjustAmp(f_amp)) << 8 & 0xFF00;
    data[dev_idx][trans_idx++] = duty | phase;
    if (trans_idx == NUM_TRANS_IN_UNIT) {
      dev_idx++;
      trans_idx = 0;
    }
  }
}

void HoloGainImplSDP(vector<AUTDDataArray>& data, const MatrixX3d& foci, const VectorXd& amps, const GeometryPtr& geometry, void* params) {
  double alpha, lambda;
  int32_t repeat;
  bool normalize;

  if (params != nullptr) {
    auto* const sdp_params = static_cast<autd::gain::SDPParams*>(params);
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

  const size_t m = foci.rows();
  const auto n = geometry->num_transducers();

  MatrixXcd p = MatrixXcd::Zero(m, m);
  for (size_t i = 0; i < m; i++) {
    p(i, i) = amps(i);
  }

  const auto b = TransferMatrix(geometry, foci, m, n);

  std::random_device seed_gen;
  std::mt19937 mt(seed_gen());
  std::uniform_real_distribution<double> range(0, 1);

  const Eigen::JacobiSVD<MatrixXcd> svd(b, Eigen::ComputeThinU | Eigen::ComputeThinV);
  auto singular_values_inv = svd.singularValues();
  for (int64_t i = 0; i < singular_values_inv.size(); ++i) {
    singular_values_inv(i) = singular_values_inv(i) / (singular_values_inv(i) * singular_values_inv(i) + alpha * alpha);
  }
  const MatrixXcd p_inv_b = svd.matrixV() * singular_values_inv.asDiagonal() * svd.matrixU().adjoint();

  MatrixXcd mm = p * (MatrixXcd::Identity(m, m) - b * p_inv_b) * p;
  MatrixXcd x_mat = MatrixXcd::Identity(m, m);
  for (auto i = 0; i < repeat; i++) {
    const auto rnd = range(mt);
    const auto ii = static_cast<size_t>(static_cast<double>(m) * rnd);

    auto xc = x_mat;
    RemoveRow(&xc, ii);
    RemoveColumn(&xc, ii);
    VectorXcd mmc = mm.col(ii);
    mmc.block(ii, 0, mmc.rows() - 1 - ii, 1) = mmc.block(ii + 1, 0, mmc.rows() - 1 - ii, 1);
    mmc.conservativeResize(mmc.rows() - 1, 1);

    VectorXcd x = xc * mmc;
    complex<double> gamma = x.adjoint() * mmc;
    if (gamma.real() > 0) {
      x = -x * sqrt(lambda / gamma.real());
      x_mat.block(ii, 0, 1, ii) = x.block(0, 0, ii, 1).adjoint().eval();
      x_mat.block(ii, ii + 1, 1, m - ii - 1) = x.block(ii, 0, m - 1 - ii, 1).adjoint().eval();
      x_mat.block(0, ii, ii, 1) = x.block(0, 0, ii, 1).eval();
      x_mat.block(ii + 1, ii, m - ii - 1, 1) = x.block(ii, 0, m - 1 - ii, 1).eval();
    } else {
      x_mat.block(ii, 0, 1, ii) = VectorXcd::Zero(ii).adjoint();
      x_mat.block(ii, ii + 1, 1, m - ii - 1) = VectorXcd::Zero(m - ii - 1).adjoint();
      x_mat.block(0, ii, ii, 1) = VectorXcd::Zero(ii);
      x_mat.block(ii + 1, ii, m - ii - 1, 1) = VectorXcd::Zero(m - ii - 1);
    }
  }

  const Eigen::ComplexEigenSolver<MatrixXcd> ces(x_mat);
  const auto& evs = ces.eigenvalues();
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
  const auto q = p_inv_b * p * u;
  const auto max_coeff = sqrt(q.cwiseAbs2().maxCoeff());

  SetFromComplexDrive(data, q, normalize, max_coeff);
}

void HoloGainImplEVD(vector<AUTDDataArray>& data, const MatrixX3d& foci, const VectorXd& amps, const GeometryPtr& geometry, void* params) {
  double gamma;
  bool normalize;

  if (params != nullptr) {
    auto* const evd_params = static_cast<autd::gain::EVDParams*>(params);
    gamma = evd_params->regularization < 0 ? 1.0 : evd_params->regularization;
    normalize = evd_params->normalize_amp;
  } else {
    gamma = 1;
    normalize = true;
  }

  const size_t m = foci.rows();
  const auto n = geometry->num_transducers();

  auto g = TransferMatrix(geometry, foci, m, n);

  VectorXcd denominator(m);
  for (size_t i = 0; i < m; i++) {
    auto tmp = complex<double>(0, 0);
    for (size_t j = 0; j < n; j++) {
      tmp += g(i, j);
    }
    denominator(i) = tmp;
  }

  MatrixXcd x(n, m);
  for (size_t i = 0; i < m; i++) {
    auto c = complex<double>(amps(i), 0) / denominator(i);
    for (size_t j = 0; j < n; j++) {
      x(j, i) = c * std::conj(g(i, j));
    }
  }
  auto r = g * x;

  Eigen::ComplexEigenSolver<MatrixXcd> ces(r);
  const auto& evs = ces.eigenvalues();
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
  VectorXd e_arg(m);
  for (size_t i = 0; i < m; i++) {
    e_arg(i) = arg(max_ev(i));
  }

  auto sigma = MatrixXcd(n, n);
  for (size_t j = 0; j < n; j++) {
    auto tmp = 0.0;
    for (size_t i = 0; i < m; i++) {
      tmp += abs(g(i, j)) * amps(i);
    }
    sigma(j, j) = complex<double>(pow(sqrt(tmp / static_cast<double>(m)), gamma), 0.0);
  }

  MatrixXcd gr(g.rows() + sigma.rows(), g.cols());
  gr << g, sigma;

  VectorXcd f = VectorXcd::Zero(m + n);
  for (size_t i = 0; i < m; i++) {
    f(i) = amps(i) * exp(complex<double>(0.0, e_arg(i)));
  }

  auto gt = gr.adjoint();
  auto gtg = gt * gr;
  auto gtf = gt * f;
  Eigen::FullPivHouseholderQR<MatrixXcd> qr(gtg);
  auto q = qr.solve(gtf);

  auto max_coeff = sqrt(q.cwiseAbs2().maxCoeff());

  SetFromComplexDrive(data, q, normalize, max_coeff);
}

void HoloGainImplNaive(vector<AUTDDataArray>& data, const MatrixX3d& foci, const VectorXd& amps, const GeometryPtr& geometry) {
  const size_t m = foci.rows();
  const auto n = geometry->num_transducers();

  const auto g = TransferMatrix(geometry, foci, m, n);
  const auto gh = g.adjoint();
  const VectorXcd p = amps;
  const auto q = gh * p;
  SetFromComplexDrive(data, q, true, 1.0);
}

void HoloGainImplGS(vector<AUTDDataArray>& data, const MatrixX3d& foci, const VectorXd& amps, const GeometryPtr& geometry, void* params) {
  const int32_t repeat = params == nullptr ? 100 : *static_cast<uint32_t*>(params);

  const size_t m = foci.rows();
  const auto n = geometry->num_transducers();

  const auto g = TransferMatrix(geometry, foci, m, n);

  const auto gh = g.adjoint();

  VectorXcd p0 = amps;
  VectorXcd q0 = VectorXcd::Ones(n);

  auto q = q0;
  for (auto k = 0; k < repeat; k++) {
    auto gamma = g * q;
    VectorXcd p(m);
    for (size_t i = 0; i < m; i++) p(i) = gamma(i) / abs(gamma(i)) * p0(i);
    auto xi = gh * p;
    for (size_t j = 0; j < n; j++) q(j) = xi(j) / abs(xi(j)) * q0(j);
  }

  SetFromComplexDrive(data, q, true, 1.0);
}

void HoloGainImplGSPAT(vector<AUTDDataArray>& data, const MatrixX3d& foci, const VectorXd& amps, const GeometryPtr& geometry, void* params) {
  const int32_t repeat = params == nullptr ? 100 : *static_cast<uint32_t*>(params);

  const size_t m = foci.rows();
  const auto n = geometry->num_transducers();

  auto g = TransferMatrix(geometry, foci, m, n);

  VectorXcd denominator(m);
  for (size_t i = 0; i < m; i++) {
    auto tmp = complex<double>(0, 0);
    for (size_t j = 0; j < n; j++) tmp += abs(g(i, j));
    denominator(i) = tmp;
  }

  MatrixXcd b(n, m);
  for (size_t i = 0; i < m; i++) {
    auto d = denominator(i) * denominator(i);
    for (size_t j = 0; j < n; j++) {
      b(j, i) = std::conj(g(i, j)) / d;
    }
  }

  const auto r = g * b;

  VectorXcd p0 = amps;
  auto p = p0;
  VectorXcd gamma = r * p;
  for (auto k = 0; k < repeat; k++) {
    for (size_t i = 0; i < m; i++) p(i) = gamma(i) / abs(gamma(i)) * p0(i);
    gamma = r * p;
  }

  for (size_t i = 0; i < m; i++) p(i) = gamma(i) / (abs(gamma(i)) * abs(gamma(i))) * p0(i) * p0(i);
  const auto q = b * p;

  SetFromComplexDrive(data, q, true, 1.0);
}

inline MatrixXcd CalcTTh(const VectorXd& x) {
  const size_t len = x.size();
  VectorXcd t(len);
  for (size_t i = 0; i < len; i++) t(i) = exp(complex<double>(0, -x(i)));
  return t * t.adjoint();
}

inline MatrixXcd MakeBhB(const GeometryPtr& geometry, const MatrixX3d& foci, const VectorXd& amps, const size_t m, const size_t n) {
  MatrixXcd p = MatrixXcd::Zero(m, m);
  for (size_t i = 0; i < m; i++) p(i, i) = -amps(i);

  const auto g = TransferMatrix(geometry, foci, m, n);

  MatrixXcd b(g.rows(), g.cols() + p.cols());
  b << g, p;
  return b.adjoint() * b;
}

void HoloGainImplLM(vector<AUTDDataArray>& data, const MatrixX3d& foci, const VectorXd& amps, const GeometryPtr& geometry, void* params) {
  double eps_1, eps_2, tau;
  int32_t k_max;

  if (params != nullptr) {
    auto* const nlp_params = static_cast<autd::gain::NLSParams*>(params);
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

  const size_t m = foci.rows();
  const auto n = geometry->num_transducers();
  const auto n_param = n + m;

  VectorXd x0 = VectorXd::Zero(n_param);
  MatrixXd identity = MatrixXd::Identity(n_param, n_param);

  auto bhb = MakeBhB(geometry, foci, amps, m, n);

  auto x = x0;
  auto nu = 2.0;

  auto tth = CalcTTh(x);
  MatrixXcd bhb_tth = bhb.cwiseProduct(tth);
  MatrixXd a = bhb_tth.real();
  VectorXd g(n_param);
  for (size_t i = 0; i < n_param; i++) {
    auto tmp = 0.0;
    for (size_t k = 0; k < n_param; k++) tmp += bhb_tth(i, k).imag();
    g(i) = tmp;
  }

  auto a_max = a.diagonal().maxCoeff();
  auto mu = tau * a_max;

  auto is_found = g.maxCoeff() <= eps_1;

  VectorXcd t(n_param);
  for (size_t i = 0; i < n_param; i++) t(i) = exp(complex<double>(0, x(i)));
  auto fx = (t.adjoint() * bhb * t)[0].real();

  for (auto k = 0; k < k_max; k++) {
    if (is_found) break;

    Eigen::HouseholderQR<MatrixXd> qr(a + mu * identity);
    auto h_lm = -qr.solve(g);
    if (h_lm.norm() <= eps_2 * (x.norm() + eps_2)) {
      is_found = true;
    } else {
      auto x_new = x + h_lm;
      for (size_t i = 0; i < n_param; i++) t(i) = exp(complex<double>(0, x_new(i)));
      auto fx_new = (t.adjoint() * bhb * t)[0].real();
      auto l0_lhlm = 0.5 * h_lm.dot(mu * h_lm - g);
      auto rho = (fx - fx_new) / l0_lhlm;
      fx = fx_new;
      if (rho > 0.0) {
        x = x_new;
        tth = CalcTTh(x);
        bhb_tth = bhb.cwiseProduct(tth);
        a = bhb_tth.real();
        for (size_t i = 0; i < n_param; i++) {
          auto tmp = 0.0;
          for (size_t j = 0; j < n_param; j++) tmp += bhb_tth(i, j).imag();
          g(i) = tmp;
        }
        is_found = g.maxCoeff() <= eps_1;
        mu *= std::max(1.0 / 3., pow(1. - (2.0 * rho - 1.), 3));
        nu = 2.0;
      } else {
        mu *= nu;
        nu *= 2.0;
      }
    }
  }

  const uint16_t duty = 0xFF00;
  size_t dev_idx = 0;
  size_t trans_idx = 0;
  for (size_t j = 0; j < n; j++) {
    const auto f_phase = fmod(x(j), 2 * M_PI) / (2 * M_PI);
    const auto phase = static_cast<uint16_t>((1 - f_phase) * 255.);
    data[dev_idx][trans_idx++] = duty | phase;
    if (trans_idx == NUM_TRANS_IN_UNIT) {
      dev_idx++;
      trans_idx = 0;
    }
  }
}
}  // namespace hologainimpl

namespace autd::gain {

GainPtr HoloGain::Create(const std::vector<utils::Vector3>& foci, const std::vector<double>& amps, const OPT_METHOD method, void* params) {
  GainPtr ptr = std::make_shared<HoloGain>(Convert(foci), amps, method, params);
  return ptr;
}
#ifdef USE_EIGEN_AUTD
GainPtr HoloGain::Create(const std::vector<Vector3>& foci, const std::vector<double>& amps, const OPT_METHOD method, void* params) {
  GainPtr ptr = std::make_shared<HoloGain>(foci, amps, method, params);
  return ptr;
}
#endif

void HoloGain::Build() {
  if (this->built()) return;
  const auto geo = this->geometry();

  CheckAndInit(geo, &this->_data);

  const auto m = _foci.size();

  Eigen::MatrixX3d foci(m, 3);
  Eigen::VectorXd amps(m);

  for (size_t i = 0; i < m; i++) {
    foci.row(i) = ConvertToEigen(_foci[i]);
    amps(i) = _amps[i];
  }

  switch (this->_method) {
    case OPT_METHOD::SDP:
      hologainimpl::HoloGainImplSDP(_data, foci, amps, geo, _params);
      break;
    case OPT_METHOD::EVD:
      hologainimpl::HoloGainImplEVD(_data, foci, amps, geo, _params);
      break;
    case OPT_METHOD::NAIVE:
      hologainimpl::HoloGainImplNaive(_data, foci, amps, geo);
      break;
    case OPT_METHOD::GS:
      hologainimpl::HoloGainImplGS(_data, foci, amps, geo, _params);
      break;
    case OPT_METHOD::GS_PAT:
      hologainimpl::HoloGainImplGSPAT(_data, foci, amps, geo, _params);
      break;
    case OPT_METHOD::LM:
      hologainimpl::HoloGainImplLM(_data, foci, amps, geo, _params);
      break;
  }
}
}  // namespace autd::gain
