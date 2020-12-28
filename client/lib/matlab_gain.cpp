// File: matlab_gain.cpp
// Project: lib
// Created Date: 20/09/2016
// Author:Seki Inoue
// -----
// Last Modified: 27/12/2020
// Modified By: Shun Suzuki (suzuki@hapis.k.u-tokyo.ac.jp)
// -----
// Copyright (c) 2016-2020 Hapis Lab. All rights reserved.
//

#ifdef MATLAB_ENABLED
#include <mat.h>
#include <matrix.h>

#include <algorithm>
#include <complex>
#include <iostream>
#endif

#include "gain.hpp"

namespace autd::gain {

GainPtr MatlabGain::Create(const std::string &filename, const std::string &var_name) {
#ifndef MATLAB_ENABLED
  throw std::runtime_error("MatlabGain requires Matlab libraries. Recompile with Matlab Environment.");
#else
  GainPtr ptr = std::make_shared<MatlabGain>(filename, var_name);
  return ptr;
#endif
}

void MatlabGain::Build() {
  if (this->built()) return;

  const auto geometry = this->geometry();

  CheckAndInit(geometry, &this->_data);

#ifdef MATLAB_ENABLED
  const auto num_trans = this->geometry()->num_transducers();

  const auto p_mat = matOpen(_filename.c_str(), "r");
  if (p_mat == nullptr) {
    throw std::runtime_error("Cannot open a file " + _filename);
  }

  const auto arr = matGetVariable(p_mat, _var_name.c_str());
  const auto num_elems = mxGetNumberOfElements(arr);
  if (num_trans < num_elems) {
    throw std::runtime_error("Insufficient number of data in mat file");
  }

  const auto array = mxGetComplexDoubles(arr);

  const auto pos = matGetVariable(p_mat, "pos");
  double *pos_arr = nullptr;
  if (pos != nullptr) {
    pos_arr = mxGetPr(pos);
  }

  for (size_t i = 0; i < num_elems; i++) {
    const auto f_amp = sqrt(array[i].real * array[i].real + array[i].imag * array[i].imag);
    const auto amp = static_cast<uint16_t>(std::clamp<double>(f_amp, 0, 1) * 255.0);
    auto f_phase = 0.0;
    if (amp != 0) f_phase = atan2(array[i].imag, array[i].real);
    const auto phase = static_cast<uint16_t>(round((-f_phase + M_PI) / (2.0 * M_PI) * 255.0));

    if (pos_arr != nullptr) {
      const auto x = static_cast<Float>(pos_arr[i * 3 + 0]);
      const auto y = static_cast<Float>(pos_arr[i * 3 + 1]);
      const auto z = static_cast<Float>(pos_arr[i * 3 + 2]);
      auto mtp = Vector3(x, y, z) * 10.0;
      auto trp = this->geometry()->position(i);
      if ((mtp - trp).norm() > 10) {
        std::cout << "Warning: position mismatch at " << i << std::endl << mtp << std::endl << trp << std::endl;
      }
    }

    const uint16_t duty = amp << 8 & 0xFF00;
    this->_data[this->geometry()->device_idx_for_trans_idx(i)][i % NUM_TRANS_IN_UNIT] = duty | phase;
  }
#endif

  this->_built = true;
}
}  // namespace autd::gain
