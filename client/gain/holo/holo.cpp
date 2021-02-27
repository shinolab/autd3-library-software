﻿// File: holo_gain.cpp
// Project: lib
// Created Date: 06/07/2016
// Author: Seki Inoue
// -----
// Last Modified: 27/02/2021
// Modified By: Shun Suzuki (suzuki@hapis.k.u-tokyo.ac.jp)
// -----
// Copyright (c) 2016-2020 Hapis Lab. All rights reserved.
//

#include "gain/holo.hpp"

namespace {

using autd::Float;
using autd::PI;

#ifdef USE_DOUBLE_AUTD
constexpr double DIR_COEFF_A[] = {1.0, 1.0, 1.0, 0.891250938, 0.707945784, 0.501187234, 0.354813389, 0.251188643, 0.199526231};
constexpr double DIR_COEFF_B[] = {
    0., 0., -0.00459648054721, -0.0155520765675, -0.0208114779827, -0.0182211227016, -0.0122437497109, -0.00780345575475, -0.00312857467007};
constexpr double DIR_COEFF_C[] = {
    0., 0., -0.000787968093807, -0.000307591508224, -0.000218348633296, 0.00047738416141, 0.000120353137658, 0.000323676257958, 0.000143850511};
constexpr double DIR_COEFF_D[] = {
    0., 0., 1.60125528528e-05, 2.9747624976e-06, 2.31910931569e-05, -1.1901034125e-05, 6.77743734332e-06, -5.99548024824e-06, -4.79372835035e-06};
using complex = std::complex<double>;
#else
constexpr float DIR_COEFF_A[] = {1.0f, 1.0f, 1.0f, 0.891250938f, 0.707945784f, 0.501187234f, 0.354813389f, 0.251188643f, 0.199526231f};
constexpr float DIR_COEFF_B[] = {
    0.f, 0.f, -0.00459648054721f, -0.0155520765675f, -0.0208114779827f, -0.0182211227016f, -0.0122437497109f, -0.00780345575475f, -0.00312857467007f};
constexpr float DIR_COEFF_C[] = {0.f,
                                 0.f,
                                 -0.000787968093807f,
                                 -0.000307591508224f,
                                 -0.000218348633296f,
                                 0.00047738416141f,
                                 0.000120353137658f,
                                 0.000323676257958f,
                                 0.000143850511f};
constexpr float DIR_COEFF_D[] = {0.f,
                                 0.f,
                                 1.60125528528e-05f,
                                 2.9747624976e-06f,
                                 2.31910931569e-05f,
                                 -1.1901034125e-05f,
                                 6.77743734332e-06f,
                                 -5.99548024824e-06f,
                                 -4.79372835035e-06f};
using complex = std::complex<float>;
#endif

static constexpr Float ATTENUATION = Float{1.15e-4};

static Float directivityT4010A1(Float theta_deg) {
  theta_deg = abs(theta_deg);

  while (theta_deg > 90) {
    theta_deg = abs(180 - theta_deg);
  }

  const auto i = static_cast<size_t>(ceil(theta_deg / 10));

  if (i == 0) {
    return 1;
  }

  const auto a = DIR_COEFF_A[i - 1];
  const auto b = DIR_COEFF_B[i - 1];
  const auto c = DIR_COEFF_C[i - 1];
  const auto d = DIR_COEFF_D[i - 1];
  const auto x = theta_deg - static_cast<Float>(i - 1) * 10;
  return a + b * x + c * x * x + d * x * x * x;
}

complex transfer(const autd::Vector3& trans_pos, const autd::Vector3& trans_norm, const autd::Vector3& target_pos, const Float wave_number) {
  const auto diff = target_pos - trans_pos;
  const auto dist = diff.norm();
  const auto theta = atan2(diff.dot(trans_norm), dist * trans_norm.norm()) * 180 / PI;
  const auto directivity = directivityT4010A1(theta);

  return directivity / dist * exp(complex(-dist * ATTENUATION, -wave_number * dist));
}
}  // namespace

namespace autd::gain {

Eigen3Backend::MatrixXc Eigen3Backend::pseudoInverseSVD(const Eigen3Backend::MatrixXc& matrix, Float alpha) {
  Eigen::JacobiSVD<MatrixXc> svd(matrix, Eigen::ComputeThinU | Eigen::ComputeThinV);
  Eigen::JacobiSVD<MatrixXc>::SingularValuesType singularValues_inv = svd.singularValues();
  for (auto i = 0; i < singularValues_inv.size(); i++) {
    singularValues_inv(i) = singularValues_inv(i) / (singularValues_inv(i) * singularValues_inv(i) + alpha);
  }
  MatrixXc pinvB = (svd.matrixV() * singularValues_inv.asDiagonal() * svd.matrixU().adjoint());
  return pinvB;
}

Eigen3Backend::MatrixXc Eigen3Backend::transferMatrix(const GeometryPtr& geometry, const std::vector<Vector3>& foci) {
  const size_t m = foci.size();
  const size_t n = geometry->num_transducers();

  auto g = Eigen3Backend::MatrixXc(m, n);

  const auto wave_number = 2 * PI / geometry->wavelength();
  for (size_t i = 0; i < m; i++) {
    const auto& tp = foci[i];
    for (size_t j = 0; j < n; j++) {
      const auto pos = geometry->position(j);
      const auto dir = geometry->direction(j / NUM_TRANS_IN_UNIT);
      g(i, j) = transfer(pos, dir, tp, wave_number);
    }
  }

  return g;
}
}  // namespace autd::gain

//#include <complex>
//#include <map>
//#include <random>
//#include <utility>
//#include <vector>
//
//#include "autd_types.hpp"
//#include "consts.hpp"
//#include "gain.hpp"

// namespace hologainimpl {

//
// void RemoveRow(MatrixXc* const matrix, const size_t row_to_remove) {
//  const auto num_rows = static_cast<size_t>(matrix->rows()) - 1;
//  const auto num_cols = static_cast<size_t>(matrix->cols());
//
//  if (row_to_remove < num_rows)
//    matrix->block(row_to_remove, 0, num_rows - row_to_remove, num_cols) = matrix->block(row_to_remove + 1, 0, num_rows - row_to_remove, num_cols);
//
//  matrix->conservativeResize(num_rows, num_cols);
//}
//
// void RemoveColumn(MatrixXc* const matrix, const size_t col_to_remove) {
//  const auto num_rows = static_cast<size_t>(matrix->rows());
//  const auto num_cols = static_cast<size_t>(matrix->cols()) - 1;
//
//  if (col_to_remove < num_cols)
//    matrix->block(0, col_to_remove, num_rows, num_cols - col_to_remove) = matrix->block(0, col_to_remove + 1, num_rows, num_cols - col_to_remove);
//
//  matrix->conservativeResize(num_rows, num_cols);
//}
//

//
// inline MatrixXc pseudoInverseSVD(const MatrixXc& matrix, Float alpha) {
//  Eigen::JacobiSVD<MatrixXc> svd(matrix, Eigen::ComputeThinU | Eigen::ComputeThinV);
//  Eigen::JacobiSVD<MatrixXc>::SingularValuesType singularValues_inv = svd.singularValues();
//  for (auto i = 0; i < singularValues_inv.size(); i++) {
//    singularValues_inv(i) = singularValues_inv(i) / (singularValues_inv(i) * singularValues_inv(i) + alpha);
//  }
//  MatrixXc pinvB = (svd.matrixV() * singularValues_inv.asDiagonal() * svd.matrixU().adjoint());
//  return pinvB;
//}
//
// inline void setBCDResult(MatrixXc* const mat, VectorXc& vec, size_t idx) {
//  const auto M = vec.size();
//  mat->block(idx, 0, 1, idx) = vec.block(0, 0, idx, 1).adjoint();
//  mat->block(idx, idx + 1, 1, M - idx - 1) = vec.block(idx + 1, 0, M - idx - 1, 1).adjoint();
//  mat->block(0, idx, idx, 1) = vec.block(0, 0, idx, 1);
//  mat->block(idx + 1, idx, M - idx - 1, 1) = vec.block(idx + 1, 0, M - idx - 1, 1);
//}
//
// void SetFromComplexDrive(vector<AUTDDataArray>& data, const VectorXc& drive, const bool normalize, const Float max_coeff) {
//  const size_t n = drive.size();
//  size_t dev_idx = 0;
//  size_t trans_idx = 0;
//  for (size_t j = 0; j < n; j++) {
//    const auto f_amp = normalize ? Float{1} : abs(drive(j)) / max_coeff;
//    const auto f_phase = arg(drive(j)) / (2 * PI) + Float{0.5};
//    const auto phase = static_cast<uint16_t>((1 - f_phase) * Float{255});
//    const uint16_t duty = static_cast<uint16_t>(ToDuty(f_amp)) << 8 & 0xFF00;
//    data[dev_idx][trans_idx++] = duty | phase;
//    if (trans_idx == NUM_TRANS_IN_UNIT) {
//      dev_idx++;
//      trans_idx = 0;
//    }
//  }
//}
//
// void HoloGainImplSDP(vector<AUTDDataArray>& data, const MatrixX3& foci, const VectorX& amps, const GeometryPtr& geometry, void* params) {
//  auto alpha = Float{1e-3};
//  auto lambda = Float{0.9};
//  auto repeat = 100;
//  auto normalize = true;
//
//  if (params != nullptr) {
//    auto* const sdp_params = static_cast<autd::gain::SDPParams*>(params);
//    alpha = sdp_params->regularization < 0 ? alpha : sdp_params->regularization;
//    repeat = sdp_params->repeat < 0 ? repeat : sdp_params->repeat;
//    lambda = sdp_params->lambda < 0 ? lambda : sdp_params->lambda;
//    normalize = sdp_params->normalize_amp;
//  }
//
//  const size_t M = foci.rows();
//  const auto N = geometry->num_transducers();
//
//  MatrixXc P = MatrixXc::Zero(M, M);
//  for (size_t i = 0; i < M; i++) P(i, i) = amps(i);
//
//  const MatrixXc B = TransferMatrix(geometry, foci, M, N);
//  MatrixXc pinvB = pseudoInverseSVD(B, alpha);
//
//  MatrixXc MM = P * (MatrixXc::Identity(M, M) - B * pinvB) * P;
//  MatrixXc X = MatrixXc::Identity(M, M);
//
//  std::random_device rnd;
//  std::mt19937 mt(rnd());
//  std::uniform_real_distribution<double> range(0, 1);
//  VectorXc zero = VectorXc::Zero(M);
//  for (auto i = 0; i < repeat; i++) {
//    auto ii = static_cast<size_t>(M * static_cast<double>(range(mt)));
//
//    VectorXc mmc = MM.col(ii);
//    mmc(ii) = 0;
//
//    VectorXc x = X * mmc;
//    complex gamma = x.adjoint() * mmc;
//    if (gamma.real() > 0) {
//      x = -x * sqrt(lambda / gamma.real());
//      setBCDResult(&X, x, ii);
//    } else {
//      setBCDResult(&X, zero, ii);
//    }
//  }
//
//  const Eigen::ComplexEigenSolver<MatrixXc> ces(X);
//  int idx = 0;
//  ces.eigenvalues().cwiseAbs2().maxCoeff(&idx);
//  VectorXc u = ces.eigenvectors().col(idx);
//
//  const auto q = pinvB * P * u;
//  const auto max_coeff = sqrt(q.cwiseAbs2().maxCoeff());
//
//  SetFromComplexDrive(data, q, normalize, max_coeff);
//}

// void HoloGainImplEVD(vector<AUTDDataArray>& data, const MatrixX3& foci, const VectorX& amps, const GeometryPtr& geometry, void* params) {
//  Float gamma = 1;
//  auto normalize = true;
//
//  if (params != nullptr) {
//    auto* const evd_params = static_cast<autd::gain::EVDParams*>(params);
//    gamma = evd_params->regularization < 0 ? gamma : evd_params->regularization;
//    normalize = evd_params->normalize_amp;
//  }
//
//  const size_t m = foci.rows();
//  const auto n = geometry->num_transducers();
//
//  auto g = TransferMatrix(geometry, foci, m, n);
//
//  VectorXc denominator(m);
//  for (size_t i = 0; i < m; i++) {
//    auto tmp = complex(0, 0);
//    for (size_t j = 0; j < n; j++) {
//      tmp += g(i, j);
//    }
//    denominator(i) = tmp;
//  }
//
//  MatrixXc x(n, m);
//  for (size_t i = 0; i < m; i++) {
//    auto c = complex(amps(i), 0) / denominator(i);
//    for (size_t j = 0; j < n; j++) {
//      x(j, i) = c * std::conj(g(i, j));
//    }
//  }
//  auto r = g * x;
//
//  Eigen::ComplexEigenSolver<MatrixXc> ces(r);
//  const auto& evs = ces.eigenvalues();
//  Float abs_eiv = 0;
//  auto idx = 0;
//  for (auto j = 0; j < evs.rows(); j++) {
//    const auto eiv = abs(evs(j));
//    if (abs_eiv < eiv) {
//      abs_eiv = eiv;
//      idx = j;
//    }
//  }
//  auto max_ev = ces.eigenvectors().row(idx);
//  VectorX e_arg(m);
//  for (size_t i = 0; i < m; i++) {
//    e_arg(i) = arg(max_ev(i));
//  }
//
//  auto sigma = MatrixXc(n, n);
//  for (size_t j = 0; j < n; j++) {
//    Float tmp = 0;
//    for (size_t i = 0; i < m; i++) {
//      tmp += abs(g(i, j)) * amps(i);
//    }
//    sigma(j, j) = complex(pow(sqrt(tmp / static_cast<Float>(m)), gamma), 0.0);
//  }
//
//  MatrixXc gr(g.rows() + sigma.rows(), g.cols());
//  gr << g, sigma;
//
//  VectorXc f = VectorXc::Zero(m + n);
//  for (size_t i = 0; i < m; i++) {
//    f(i) = amps(i) * exp(complex(0.0, e_arg(i)));
//  }
//
//  auto gt = gr.adjoint();
//  auto gtg = gt * gr;
//  auto gtf = gt * f;
//  Eigen::FullPivHouseholderQR<MatrixXc> qr(gtg);
//  auto q = qr.solve(gtf);
//
//  auto max_coeff = sqrt(q.cwiseAbs2().maxCoeff());
//
//  SetFromComplexDrive(data, q, normalize, max_coeff);
//}
//
// void HoloGainImplNaive(vector<AUTDDataArray>& data, const MatrixX3& foci, const VectorX& amps, const GeometryPtr& geometry) {
//  const size_t m = foci.rows();
//  const auto n = geometry->num_transducers();
//
//  const auto g = TransferMatrix(geometry, foci, m, n);
//  const auto gh = g.adjoint();
//  const VectorXc p = amps;
//  const auto q = gh * p;
//  SetFromComplexDrive(data, q, true, 1.0);
//}
//
// void HoloGainImplGS(vector<AUTDDataArray>& data, const MatrixX3& foci, const VectorX& amps, const GeometryPtr& geometry, void* params) {
//  const int32_t repeat = params == nullptr ? 100 : *static_cast<uint32_t*>(params);
//
//  const size_t m = foci.rows();
//  const auto n = geometry->num_transducers();
//
//  const auto g = TransferMatrix(geometry, foci, m, n);
//
//  const auto gh = g.adjoint();
//
//  VectorXc p0 = amps;
//  VectorXc q0 = VectorXc::Ones(n);
//
//  VectorXc q = q0;
//  for (auto k = 0; k < repeat; k++) {
//    auto gamma = g * q;
//    VectorXc p(m);
//    for (size_t i = 0; i < m; i++) p(i) = gamma(i) / abs(gamma(i)) * p0(i);
//    auto xi = gh * p;
//    for (size_t j = 0; j < n; j++) q(j) = xi(j) / abs(xi(j)) * q0(j);
//  }
//
//  SetFromComplexDrive(data, q, true, 1.0);
//}
//
// void HoloGainImplGSPAT(vector<AUTDDataArray>& data, const MatrixX3& foci, const VectorX& amps, const GeometryPtr& geometry, void* params) {
//  const int32_t repeat = params == nullptr ? 100 : *static_cast<uint32_t*>(params);
//
//  const size_t m = foci.rows();
//  const auto n = geometry->num_transducers();
//
//  auto g = TransferMatrix(geometry, foci, m, n);
//
//  VectorXc denominator(m);
//  for (size_t i = 0; i < m; i++) {
//    auto tmp = complex(0, 0);
//    for (size_t j = 0; j < n; j++) tmp += abs(g(i, j));
//    denominator(i) = tmp;
//  }
//
//  MatrixXc b(n, m);
//  for (size_t i = 0; i < m; i++) {
//    auto d = denominator(i) * denominator(i);
//    for (size_t j = 0; j < n; j++) {
//      b(j, i) = std::conj(g(i, j)) / d;
//    }
//  }
//
//  const auto r = g * b;
//
//  VectorXc p0 = amps;
//  VectorXc p = p0;
//  VectorXc gamma = r * p;
//  for (auto k = 0; k < repeat; k++) {
//    for (size_t i = 0; i < m; i++) p(i) = gamma(i) / abs(gamma(i)) * p0(i);
//    gamma = r * p;
//  }
//
//  for (size_t i = 0; i < m; i++) p(i) = gamma(i) / (abs(gamma(i)) * abs(gamma(i))) * p0(i) * p0(i);
//  const auto q = b * p;
//
//  SetFromComplexDrive(data, q, true, 1.0);
//}
//
// inline MatrixXc CalcTTh(const VectorX& x) {
//  const size_t len = x.size();
//  VectorXc t(len);
//  for (size_t i = 0; i < len; i++) t(i) = exp(complex(0, -x(i)));
//  return t * t.adjoint();
//}
//
// inline MatrixXc MakeBhB(const GeometryPtr& geometry, const MatrixX3& foci, const VectorX& amps, const size_t m, const size_t n) {
//  MatrixXc p = MatrixXc::Zero(m, m);
//  for (size_t i = 0; i < m; i++) p(i, i) = -amps(i);
//
//  const auto g = TransferMatrix(geometry, foci, m, n);
//
//  MatrixXc b(g.rows(), g.cols() + p.cols());
//  b << g, p;
//  return b.adjoint() * b;
//}
//
// void HoloGainImplLM(vector<AUTDDataArray>& data, const MatrixX3& foci, const VectorX& amps, const GeometryPtr& geometry, void* params) {
//  auto eps_1 = Float{1e-8};
//  auto eps_2 = Float{1e-8};
//  auto tau = Float{1e-3};
//  auto k_max = 5;
//
//  if (params != nullptr) {
//    auto* const nlp_params = static_cast<autd::gain::NLSParams*>(params);
//    eps_1 = nlp_params->eps_1 < 0 ? eps_1 : nlp_params->eps_1;
//    eps_2 = nlp_params->eps_2 < 0 ? eps_2 : nlp_params->eps_2;
//    k_max = nlp_params->k_max < 0 ? k_max : nlp_params->k_max;
//    tau = nlp_params->tau < 0 ? tau : nlp_params->tau;
//  }
//
//  const size_t m = foci.rows();
//  const auto n = geometry->num_transducers();
//  const auto n_param = n + m;
//
//  VectorX x0 = VectorX::Zero(n_param);
//  MatrixX identity = MatrixX::Identity(n_param, n_param);
//
//  auto bhb = MakeBhB(geometry, foci, amps, m, n);
//
//  VectorX x = x0;
//  Float nu = 2;
//
//  auto tth = CalcTTh(x);
//  MatrixXc bhb_tth = bhb.cwiseProduct(tth);
//  MatrixX a = bhb_tth.real();
//  VectorX g(n_param);
//  for (size_t i = 0; i < n_param; i++) {
//    Float tmp = 0;
//    for (size_t k = 0; k < n_param; k++) tmp += bhb_tth(i, k).imag();
//    g(i) = tmp;
//  }
//
//  auto a_max = a.diagonal().maxCoeff();
//  auto mu = tau * a_max;
//
//  auto is_found = g.maxCoeff() <= eps_1;
//
//  VectorXc t(n_param);
//  for (size_t i = 0; i < n_param; i++) t(i) = exp(complex(0, x(i)));
//  auto fx = (t.adjoint() * bhb * t)[0].real();
//
//  for (auto k = 0; k < k_max; k++) {
//    if (is_found) break;
//
//    Eigen::HouseholderQR<MatrixX> qr(a + mu * identity);
//    auto h_lm = -qr.solve(g);
//    if (h_lm.norm() <= eps_2 * (x.norm() + eps_2)) {
//      is_found = true;
//    } else {
//      auto x_new = x + h_lm;
//      for (size_t i = 0; i < n_param; i++) t(i) = exp(complex(0, x_new(i)));
//      auto fx_new = (t.adjoint() * bhb * t)[0].real();
//      auto l0_lhlm = h_lm.dot(mu * h_lm - g) / 2;
//      auto rho = (fx - fx_new) / l0_lhlm;
//      fx = fx_new;
//      if (rho > 0) {
//        x = x_new;
//        tth = CalcTTh(x);
//        bhb_tth = bhb.cwiseProduct(tth);
//        a = bhb_tth.real();
//        for (size_t i = 0; i < n_param; i++) {
//          Float tmp = 0;
//          for (size_t j = 0; j < n_param; j++) tmp += bhb_tth(i, j).imag();
//          g(i) = tmp;
//        }
//        is_found = g.maxCoeff() <= eps_1;
//        mu *= std::max(Float{1. / 3.}, pow(1 - (2 * rho - 1), Float{3.}));
//        nu = 2.0;
//      } else {
//        mu *= nu;
//        nu *= 2;
//      }
//    }
//  }
//
//  const uint16_t duty = 0xFF00;
//  size_t dev_idx = 0;
//  size_t trans_idx = 0;
//  for (size_t j = 0; j < n; j++) {
//    const auto f_phase = fmod(x(j), 2 * M_PI) / (2 * M_PI);
//    const auto phase = static_cast<uint16_t>((1 - f_phase) * 255.);
//    data[dev_idx][trans_idx++] = duty | phase;
//    if (trans_idx == NUM_TRANS_IN_UNIT) {
//      dev_idx++;
//      trans_idx = 0;
//    }
//  }
//}
//}  // namespace hologainimpl
