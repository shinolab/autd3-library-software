// File: holo_gain.cpp
// Project: lib
// Created Date: 06/07/2016
// Author: Seki Inoue
// -----
// Last Modified: 28/02/2021
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

Eigen3Backend::VectorXc Eigen3Backend::maxEigenVector(const Eigen3Backend::MatrixXc& matrix) {
  const Eigen::ComplexEigenSolver<MatrixXc> ces(matrix);
  int idx = 0;
  ces.eigenvalues().cwiseAbs2().maxCoeff(&idx);
  return ces.eigenvectors().col(idx);
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
    (*c).noalias() += alpha * (a * b);
  }
}

void Eigen3Backend::matvecmul(const char* transa, std::complex<Float> alpha, const Eigen3Backend::MatrixXc& a, const Eigen3Backend::VectorXc& b,
                              std::complex<Float> beta, Eigen3Backend::VectorXc* c) {
  *c *= beta;
  (*c).noalias() += alpha * (a * b);

  if (strcmp(transa, "C") == 0) {
    (*c).noalias() += alpha * (a.adjoint() * b);
  } else if (strcmp(transa, "T") == 0) {
    (*c).noalias() += alpha * (a.transpose() * b);
  } else {
    (*c).noalias() += alpha * (a * b);
  }
}

void Eigen3Backend::solve(const Eigen3Backend::MatrixXc& a, const Eigen3Backend::VectorXc& b, Eigen3Backend::VectorXc* c) {
  Eigen::FullPivHouseholderQR<Eigen3Backend::MatrixXc> qr(a);
  *c = qr.solve(b);
}

std::complex<Float> Eigen3Backend::dot(const Eigen3Backend::VectorXc& a, const Eigen3Backend::VectorXc& b) { return a.dot(b); }

Float Eigen3Backend::maxCoeff(const Eigen3Backend::VectorXc& v) { return sqrt(v.cwiseAbs2().maxCoeff()); }

void Eigen3Backend::concat_in_row(const Eigen3Backend::MatrixXc& a, const Eigen3Backend::MatrixXc& b, Eigen3Backend::MatrixXc* c) { *c << a, b; }

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
