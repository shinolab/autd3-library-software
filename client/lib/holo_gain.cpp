// File: holo_gain.cpp
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

#include "autd_types.hpp"
#include "consts.hpp"
#include "convert.hpp"
#include "core.hpp"
#include "gain.hpp"

namespace hologainimpl {
using autd::AUTDDataArray;
using autd::Float, autd::ToFloat;
using autd::GeometryPtr;
using autd::NUM_TRANS_IN_UNIT;
using autd::PI;
using autd::gain::AdjustAmp;
using std::map, std::vector;

#ifdef USE_DOUBLE_AUTD
constexpr double DIR_COEFF_A[] = {1.0, 1.0, 1.0, 0.891250938, 0.707945784, 0.501187234, 0.354813389, 0.251188643, 0.199526231};
constexpr double DIR_COEFF_B[] = {
    0., 0., -0.00459648054721, -0.0155520765675, -0.0208114779827, -0.0182211227016, -0.0122437497109, -0.00780345575475, -0.00312857467007};
constexpr double DIR_COEFF_C[] = {
    0., 0., -0.000787968093807, -0.000307591508224, -0.000218348633296, 0.00047738416141, 0.000120353137658, 0.000323676257958, 0.000143850511};
constexpr double DIR_COEFF_D[] = {
    0., 0., 1.60125528528e-05, 2.9747624976e-06, 2.31910931569e-05, -1.1901034125e-05, 6.77743734332e-06, -5.99548024824e-06, -4.79372835035e-06};
using MatrixX3 = Eigen::MatrixX3d;
using MatrixX = Eigen::MatrixXd;
using MatrixXc = Eigen::MatrixXcd;
using Vector3 = Eigen::Vector3d;
using VectorXc = Eigen::VectorXcd;
using VectorX = Eigen::VectorXd;
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
using MatrixX3 = Eigen::MatrixX3f;
using MatrixX = Eigen::MatrixXf;
using MatrixXc = Eigen::MatrixXcf;
using Vector3 = Eigen::Vector3f;
using VectorXc = Eigen::VectorXcf;
using VectorX = Eigen::VectorXf;
using complex = std::complex<float>;
#endif

static constexpr Float ATTENUATION = ToFloat(1.15e-4);

static Float DirectivityT4010A1(Float theta_deg) {
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

complex Transfer(const autd::Vector3& trans_pos, const autd::Vector3& trans_norm, const autd::Vector3& target_pos, const Float wave_number) {
  const auto diff = target_pos - trans_pos;
  const auto dist = diff.norm();
  const auto theta = atan2(diff.dot(trans_norm), dist * trans_norm.norm()) * 180 / PI;
  const auto directivity = DirectivityT4010A1(theta);

  return directivity / dist * exp(complex(-dist * ATTENUATION, -wave_number * dist));
}

void RemoveRow(MatrixXc* const matrix, const size_t row_to_remove) {
  const auto num_rows = static_cast<size_t>(matrix->rows()) - 1;
  const auto num_cols = static_cast<size_t>(matrix->cols());

  if (row_to_remove < num_rows)
    matrix->block(row_to_remove, 0, num_rows - row_to_remove, num_cols) = matrix->block(row_to_remove + 1, 0, num_rows - row_to_remove, num_cols);

  matrix->conservativeResize(num_rows, num_cols);
}

void RemoveColumn(MatrixXc* const matrix, const size_t col_to_remove) {
  const auto num_rows = static_cast<size_t>(matrix->rows());
  const auto num_cols = static_cast<size_t>(matrix->cols()) - 1;

  if (col_to_remove < num_cols)
    matrix->block(0, col_to_remove, num_rows, num_cols - col_to_remove) = matrix->block(0, col_to_remove + 1, num_rows, num_cols - col_to_remove);

  matrix->conservativeResize(num_rows, num_cols);
}

MatrixXc TransferMatrix(const GeometryPtr& geometry, const MatrixX3& foci, const size_t m, const size_t n) {
  auto g = MatrixXc(m, n);

  const auto wave_number = 2 * PI / geometry->wavelength();
  for (size_t i = 0; i < m; i++) {
    const auto tp = foci.row(i);
    for (size_t j = 0; j < n; j++) {
      const auto pos = geometry->position(j);
      const auto dir = geometry->direction(j / NUM_TRANS_IN_UNIT);
      g(i, j) = Transfer(autd::Convert(pos), autd::Convert(dir), autd::Convert(tp), wave_number);
    }
  }

  return g;
}

void SetFromComplexDrive(vector<AUTDDataArray>& data, const VectorXc& drive, const bool normalize, const Float max_coeff) {
  const size_t n = drive.size();
  size_t dev_idx = 0;
  size_t trans_idx = 0;
  for (size_t j = 0; j < n; j++) {
    const auto f_amp = normalize ? ToFloat(1.0) : abs(drive(j)) / max_coeff;
    const auto f_phase = arg(drive(j)) / (2 * PI) + ToFloat(0.5);
    const auto phase = static_cast<uint16_t>((1 - f_phase) * ToFloat(255.));
    const uint16_t duty = static_cast<uint16_t>(AdjustAmp(f_amp)) << 8 & 0xFF00;
    data[dev_idx][trans_idx++] = duty | phase;
    if (trans_idx == NUM_TRANS_IN_UNIT) {
      dev_idx++;
      trans_idx = 0;
    }
  }
}

void HoloGainImplSDP(vector<AUTDDataArray>& data, const MatrixX3& foci, const VectorX& amps, const GeometryPtr& geometry, void* params) {
  auto alpha = ToFloat(1e-3);
  auto lambda = ToFloat(0.8);
  auto repeat = 10;
  auto normalize = true;

  if (params != nullptr) {
    auto* const sdp_params = static_cast<autd::gain::SDPParams*>(params);
    alpha = sdp_params->regularization < 0 ? alpha : sdp_params->regularization;
    repeat = sdp_params->repeat < 0 ? repeat : sdp_params->repeat;
    lambda = sdp_params->lambda < 0 ? lambda : sdp_params->lambda;
    normalize = sdp_params->normalize_amp;
  }

  const size_t m = foci.rows();
  const auto n = geometry->num_transducers();

  MatrixXc p = MatrixXc::Zero(m, m);
  for (size_t i = 0; i < m; i++) p(i, i) = amps(i);

  const auto b = TransferMatrix(geometry, foci, m, n);

  std::random_device seed_gen;
  std::mt19937 mt(seed_gen());
  std::uniform_real_distribution<Float> range(0, 1);

  const Eigen::JacobiSVD<MatrixXc> svd(b, Eigen::ComputeThinU | Eigen::ComputeThinV);
  MatrixXc singular_values_inv = svd.singularValues();
  for (int64_t i = 0; i < singular_values_inv.size(); ++i) {
    singular_values_inv(i) = singular_values_inv(i) / (singular_values_inv(i) * singular_values_inv(i) + alpha * alpha);
  }
  const MatrixXc p_inv_b = svd.matrixV() * singular_values_inv.asDiagonal() * svd.matrixU().adjoint();

  MatrixXc mm = p * (MatrixXc::Identity(m, m) - b * p_inv_b) * p;
  MatrixXc x_mat = MatrixXc::Identity(m, m);
  for (auto i = 0; i < repeat; i++) {
    const auto rnd = range(mt);
    const auto ii = static_cast<size_t>(static_cast<Float>(m) * rnd);

    MatrixXc xc = x_mat;
    RemoveRow(&xc, ii);
    RemoveColumn(&xc, ii);
    VectorXc mmc = mm.col(ii);
    mmc.block(ii, 0, mmc.rows() - 1 - ii, 1) = mmc.block(ii + 1, 0, mmc.rows() - 1 - ii, 1);
    mmc.conservativeResize(mmc.rows() - 1, 1);

    VectorXc x = xc * mmc;
    complex gamma = x.adjoint() * mmc;
    if (gamma.real() > 0) {
      x = -x * sqrt(lambda / gamma.real());
      x_mat.block(ii, 0, 1, ii) = x.block(0, 0, ii, 1).adjoint().eval();
      x_mat.block(ii, ii + 1, 1, m - ii - 1) = x.block(ii, 0, m - 1 - ii, 1).adjoint().eval();
      x_mat.block(0, ii, ii, 1) = x.block(0, 0, ii, 1).eval();
      x_mat.block(ii + 1, ii, m - ii - 1, 1) = x.block(ii, 0, m - 1 - ii, 1).eval();
    } else {
      x_mat.block(ii, 0, 1, ii) = VectorXc::Zero(ii).adjoint();
      x_mat.block(ii, ii + 1, 1, m - ii - 1) = VectorXc::Zero(m - ii - 1).adjoint();
      x_mat.block(0, ii, ii, 1) = VectorXc::Zero(ii);
      x_mat.block(ii + 1, ii, m - ii - 1, 1) = VectorXc::Zero(m - ii - 1);
    }
  }

  const Eigen::ComplexEigenSolver<MatrixXc> ces(x_mat);
  const auto& evs = ces.eigenvalues();
  Float abs_eiv = 0;
  auto idx = 0;
  for (auto j = 0; j < evs.rows(); j++) {
    const auto eiv = abs(evs(j));
    if (abs_eiv < eiv) {
      abs_eiv = eiv;
      idx = j;
    }
  }

  const VectorXc u = ces.eigenvectors().col(idx);
  const auto q = p_inv_b * p * u;
  const auto max_coeff = sqrt(q.cwiseAbs2().maxCoeff());

  SetFromComplexDrive(data, q, normalize, max_coeff);
}

void HoloGainImplEVD(vector<AUTDDataArray>& data, const MatrixX3& foci, const VectorX& amps, const GeometryPtr& geometry, void* params) {
  Float gamma = 1;
  auto normalize = true;

  if (params != nullptr) {
    auto* const evd_params = static_cast<autd::gain::EVDParams*>(params);
    gamma = evd_params->regularization < 0 ? gamma : evd_params->regularization;
    normalize = evd_params->normalize_amp;
  }

  const size_t m = foci.rows();
  const auto n = geometry->num_transducers();

  auto g = TransferMatrix(geometry, foci, m, n);

  VectorXc denominator(m);
  for (size_t i = 0; i < m; i++) {
    auto tmp = complex(0, 0);
    for (size_t j = 0; j < n; j++) {
      tmp += g(i, j);
    }
    denominator(i) = tmp;
  }

  MatrixXc x(n, m);
  for (size_t i = 0; i < m; i++) {
    auto c = complex(amps(i), 0) / denominator(i);
    for (size_t j = 0; j < n; j++) {
      x(j, i) = c * std::conj(g(i, j));
    }
  }
  auto r = g * x;

  Eigen::ComplexEigenSolver<MatrixXc> ces(r);
  const auto& evs = ces.eigenvalues();
  Float abs_eiv = 0;
  auto idx = 0;
  for (auto j = 0; j < evs.rows(); j++) {
    const auto eiv = abs(evs(j));
    if (abs_eiv < eiv) {
      abs_eiv = eiv;
      idx = j;
    }
  }
  auto max_ev = ces.eigenvectors().row(idx);
  VectorX e_arg(m);
  for (size_t i = 0; i < m; i++) {
    e_arg(i) = arg(max_ev(i));
  }

  auto sigma = MatrixXc(n, n);
  for (size_t j = 0; j < n; j++) {
    Float tmp = 0;
    for (size_t i = 0; i < m; i++) {
      tmp += abs(g(i, j)) * amps(i);
    }
    sigma(j, j) = complex(pow(sqrt(tmp / static_cast<Float>(m)), gamma), 0.0);
  }

  MatrixXc gr(g.rows() + sigma.rows(), g.cols());
  gr << g, sigma;

  VectorXc f = VectorXc::Zero(m + n);
  for (size_t i = 0; i < m; i++) {
    f(i) = amps(i) * exp(complex(0.0, e_arg(i)));
  }

  auto gt = gr.adjoint();
  auto gtg = gt * gr;
  auto gtf = gt * f;
  Eigen::FullPivHouseholderQR<MatrixXc> qr(gtg);
  auto q = qr.solve(gtf);

  auto max_coeff = sqrt(q.cwiseAbs2().maxCoeff());

  SetFromComplexDrive(data, q, normalize, max_coeff);
}

void HoloGainImplNaive(vector<AUTDDataArray>& data, const MatrixX3& foci, const VectorX& amps, const GeometryPtr& geometry) {
  const size_t m = foci.rows();
  const auto n = geometry->num_transducers();

  const auto g = TransferMatrix(geometry, foci, m, n);
  const auto gh = g.adjoint();
  const VectorXc p = amps;
  const auto q = gh * p;
  SetFromComplexDrive(data, q, true, 1.0);
}

void HoloGainImplGS(vector<AUTDDataArray>& data, const MatrixX3& foci, const VectorX& amps, const GeometryPtr& geometry, void* params) {
  const int32_t repeat = params == nullptr ? 100 : *static_cast<uint32_t*>(params);

  const size_t m = foci.rows();
  const auto n = geometry->num_transducers();

  const auto g = TransferMatrix(geometry, foci, m, n);

  const auto gh = g.adjoint();

  VectorXc p0 = amps;
  VectorXc q0 = VectorXc::Ones(n);

  VectorXc q = q0;
  for (auto k = 0; k < repeat; k++) {
    auto gamma = g * q;
    VectorXc p(m);
    for (size_t i = 0; i < m; i++) p(i) = gamma(i) / abs(gamma(i)) * p0(i);
    auto xi = gh * p;
    for (size_t j = 0; j < n; j++) q(j) = xi(j) / abs(xi(j)) * q0(j);
  }

  SetFromComplexDrive(data, q, true, 1.0);
}

void HoloGainImplGSPAT(vector<AUTDDataArray>& data, const MatrixX3& foci, const VectorX& amps, const GeometryPtr& geometry, void* params) {
  const int32_t repeat = params == nullptr ? 100 : *static_cast<uint32_t*>(params);

  const size_t m = foci.rows();
  const auto n = geometry->num_transducers();

  auto g = TransferMatrix(geometry, foci, m, n);

  VectorXc denominator(m);
  for (size_t i = 0; i < m; i++) {
    auto tmp = complex(0, 0);
    for (size_t j = 0; j < n; j++) tmp += abs(g(i, j));
    denominator(i) = tmp;
  }

  MatrixXc b(n, m);
  for (size_t i = 0; i < m; i++) {
    auto d = denominator(i) * denominator(i);
    for (size_t j = 0; j < n; j++) {
      b(j, i) = std::conj(g(i, j)) / d;
    }
  }

  const auto r = g * b;

  VectorXc p0 = amps;
  VectorXc p = p0;
  VectorXc gamma = r * p;
  for (auto k = 0; k < repeat; k++) {
    for (size_t i = 0; i < m; i++) p(i) = gamma(i) / abs(gamma(i)) * p0(i);
    gamma = r * p;
  }

  for (size_t i = 0; i < m; i++) p(i) = gamma(i) / (abs(gamma(i)) * abs(gamma(i))) * p0(i) * p0(i);
  const auto q = b * p;

  SetFromComplexDrive(data, q, true, 1.0);
}

inline MatrixXc CalcTTh(const VectorX& x) {
  const size_t len = x.size();
  VectorXc t(len);
  for (size_t i = 0; i < len; i++) t(i) = exp(complex(0, -x(i)));
  return t * t.adjoint();
}

inline MatrixXc MakeBhB(const GeometryPtr& geometry, const MatrixX3& foci, const VectorX& amps, const size_t m, const size_t n) {
  MatrixXc p = MatrixXc::Zero(m, m);
  for (size_t i = 0; i < m; i++) p(i, i) = -amps(i);

  const auto g = TransferMatrix(geometry, foci, m, n);

  MatrixXc b(g.rows(), g.cols() + p.cols());
  b << g, p;
  return b.adjoint() * b;
}

void HoloGainImplLM(vector<AUTDDataArray>& data, const MatrixX3& foci, const VectorX& amps, const GeometryPtr& geometry, void* params) {
  auto eps_1 = ToFloat(1e-8);
  auto eps_2 = ToFloat(1e-8);
  auto tau = ToFloat(1e-3);
  auto k_max = 5;

  if (params != nullptr) {
    auto* const nlp_params = static_cast<autd::gain::NLSParams*>(params);
    eps_1 = nlp_params->eps_1 < 0 ? eps_1 : nlp_params->eps_1;
    eps_2 = nlp_params->eps_2 < 0 ? eps_2 : nlp_params->eps_2;
    k_max = nlp_params->k_max < 0 ? k_max : nlp_params->k_max;
    tau = nlp_params->tau < 0 ? tau : nlp_params->tau;
  }

  const size_t m = foci.rows();
  const auto n = geometry->num_transducers();
  const auto n_param = n + m;

  VectorX x0 = VectorX::Zero(n_param);
  MatrixX identity = MatrixX::Identity(n_param, n_param);

  auto bhb = MakeBhB(geometry, foci, amps, m, n);

  VectorX x = x0;
  Float nu = 2;

  auto tth = CalcTTh(x);
  MatrixXc bhb_tth = bhb.cwiseProduct(tth);
  MatrixX a = bhb_tth.real();
  VectorX g(n_param);
  for (size_t i = 0; i < n_param; i++) {
    Float tmp = 0;
    for (size_t k = 0; k < n_param; k++) tmp += bhb_tth(i, k).imag();
    g(i) = tmp;
  }

  auto a_max = a.diagonal().maxCoeff();
  auto mu = tau * a_max;

  auto is_found = g.maxCoeff() <= eps_1;

  VectorXc t(n_param);
  for (size_t i = 0; i < n_param; i++) t(i) = exp(complex(0, x(i)));
  auto fx = (t.adjoint() * bhb * t)[0].real();

  for (auto k = 0; k < k_max; k++) {
    if (is_found) break;

    Eigen::HouseholderQR<MatrixX> qr(a + mu * identity);
    auto h_lm = -qr.solve(g);
    if (h_lm.norm() <= eps_2 * (x.norm() + eps_2)) {
      is_found = true;
    } else {
      auto x_new = x + h_lm;
      for (size_t i = 0; i < n_param; i++) t(i) = exp(complex(0, x_new(i)));
      auto fx_new = (t.adjoint() * bhb * t)[0].real();
      auto l0_lhlm = h_lm.dot(mu * h_lm - g) / 2;
      auto rho = (fx - fx_new) / l0_lhlm;
      fx = fx_new;
      if (rho > 0) {
        x = x_new;
        tth = CalcTTh(x);
        bhb_tth = bhb.cwiseProduct(tth);
        a = bhb_tth.real();
        for (size_t i = 0; i < n_param; i++) {
          Float tmp = 0;
          for (size_t j = 0; j < n_param; j++) tmp += bhb_tth(i, j).imag();
          g(i) = tmp;
        }
        is_found = g.maxCoeff() <= eps_1;
        mu *= std::max(ToFloat(1. / 3.), pow(1 - (2 * rho - 1), ToFloat(3.)));
        nu = 2.0;
      } else {
        mu *= nu;
        nu *= 2;
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

GainPtr HoloGain::Create(const std::vector<Vector3>& foci, const std::vector<Float>& amps, const OPT_METHOD method, void* params) {
  GainPtr ptr = std::make_shared<HoloGain>(Convert(foci), amps, method, params);
  return ptr;
}

void HoloGain::Build() {
  if (this->built()) return;
  const auto geo = this->geometry();

  CheckAndInit(geo, &this->_data);

  const auto m = _foci.size();

  hologainimpl::MatrixX3 foci(m, 3);
  hologainimpl::VectorX amps(m);

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
