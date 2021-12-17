// File: holo.cpp
// Project: holo
// Created Date: 09/12/2021
// Author: Shun Suzuki
// -----
// Last Modified: 09/12/2021
// Modified By: Shun Suzuki (suzuki@hapis.k.u-tokyo.ac.jp)
// -----
// Copyright (c) 2021 Hapis Lab. All rights reserved.
//

#include "autd3/gain/holo.hpp"

namespace autd::gain::holo {
void SDP::calc(const core::Geometry& geometry) {
  this->_backend->sdp(geometry, this->_foci, this->_amps, _alpha, _lambda, _repeat, _normalize, this->_data);
}

void EVD::calc(const core::Geometry& geometry) { this->_backend->evd(geometry, this->_foci, this->_amps, _gamma, _normalize, this->_data); }

void Naive::calc(const core::Geometry& geometry) { this->_backend->naive(geometry, this->_foci, this->_amps, this->_data); }

void GS::calc(const core::Geometry& geometry) { this->_backend->gs(geometry, this->_foci, this->_amps, _repeat, this->_data); }

void GSPAT::calc(const core::Geometry& geometry) { this->_backend->gspat(geometry, this->_foci, this->_amps, _repeat, this->_data); }

void LM::calc(const core::Geometry& geometry) {
  this->_backend->lm(geometry, this->_foci, this->_amps, _eps_1, _eps_2, _tau, _k_max, _initial, this->_data);
}

void GaussNewton::calc(const core::Geometry& geometry) {
  this->_backend->gauss_newton(geometry, this->_foci, this->_amps, _eps_1, _eps_2, _k_max, _initial, this->_data);
}

void GradientDescent::calc(const core::Geometry& geometry) {
  this->_backend->gradient_descent(geometry, this->_foci, this->_amps, _eps, _step, _k_max, _initial, this->_data);
}

void APO::calc(const core::Geometry& geometry) { this->_backend->apo(geometry, this->_foci, this->_amps, _eps, _lambda, _k_max, this->_data); }

void Greedy::calc(const core::Geometry& geometry) { this->_backend->greedy(geometry, this->_foci, this->_amps, _phase_div, this->_data); }

}  // namespace autd::gain::holo
