// File: matlab_gain.cpp
// Project: lib
// Created Date: 20/09/2016
// Author:Seki Inoue
// -----
// Last Modified: 14/04/2021
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

#include "gain/matlab.hpp"

namespace autd::gain {

Result<GainPtr, std::string> MatlabGain::Create(const std::string& filename, const std::string& var_name) {
#ifndef MATLAB_ENABLED
  (void)filename;
  (void)var_name;
  return Err(std::string("MatlabGain requires Matlab libraries. Recompile with Matlab Environment."));
#else
  const GainPtr ptr = std::make_shared<MatlabGain>(filename, var_name);
  return Ok(ptr);
#endif
}

Result<bool, std::string> MatlabGain::Build() {
  if (this->built()) return Ok(false);

  const auto geometry = this->geometry();

  CheckAndInit(geometry, &this->_data);

#ifdef MATLAB_ENABLED
  const auto num_trans = this->geometry()->num_transducers();

  auto* const p_mat = matOpen(_filename.c_str(), "r");
  if (p_mat == nullptr) return Err(std::string("Cannot open a file " + _filename));

  auto* const arr = matGetVariable(p_mat, _var_name.c_str());
  const auto num_elems = mxGetNumberOfElements(arr);
  if (num_trans < num_elems) return Err(std::string("Insufficient number of data in mat file"));

  auto* const array = mxGetComplexDoubles(arr);

  auto* const pos = matGetVariable(p_mat, "pos");
  double* pos_arr = nullptr;
  if (pos != nullptr) {
    pos_arr = mxGetPr(pos);
  }

  for (size_t i = 0; i < num_elems; i++) {
    const auto re = static_cast<Float>(array[i].real);
    const auto im = static_cast<Float>(array[i].imag);
    const auto f_amp = std::sqrt(re * re + im * im);
    const auto amp = static_cast<uint16_t>(std::clamp<Float>(f_amp, 0, 1) * 255);
    Float f_phase = 0;
    if (amp != 0) f_phase = std::atan2(im, re);
    const auto phase = static_cast<uint16_t>(round((-f_phase + PI) / (2 * PI) * 255));

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
  return Ok(true);
}
}  // namespace autd::gain
