// File: matlabgain.cpp
// Project: lib
// Created Date: 20/09/2016
// Author:Seki Inoue
// -----
// Last Modified: 22/12/2020
// Modified By: Shun Suzuki (suzuki@hapis.k.u-tokyo.ac.jp)
// -----
// Copyright (c) 2016-2020 Hapis Lab. All rights reserved.
//

#ifdef MATLAB_ENABLED
#include <mat.h>
#include <matrix.h>
#endif

#define _USE_MATH_DEFINES
#include <stdio.h>

#include <algorithm>
#include <complex>
#include <iostream>

#include "consts.hpp"
#include "gain.hpp"
#include "privdef.hpp"

namespace autd::gain {

GainPtr MatlabGain::Create(std::string filename, std::string varname) {
#ifndef MATLAB_ENABLED
  throw new std::runtime_error(
      "MatlabGain requires Matlab libraries. Recompile with Matlab "
      "Environment.");
#endif
  auto ptr = std::make_shared<MatlabGain>();
  ptr->_filename = filename;
  ptr->_varname = varname;
  return ptr;
}

void MatlabGain::Build() {
  if (this->built()) return;

  auto geometry = this->geometry();

  CheckAndInit(geometry, &this->_data);

#ifdef MATLAB_ENABLED
  const int ntrans = this->geometry()->numTransducers();

  MATFile *pmat = matOpen(_filename.c_str(), "r");
  if (pmat == NULL) {
    throw new std::runtime_error("Cannot open a file " + _filename);
  }

  mxArray *arr = matGetVariable(pmat, _varname.c_str());
  size_t nelems = mxGetNumberOfElements(arr);
  if (ntrans < nelems) {
    throw new std::runtime_error("Insufficient number of data in mat file");
  }

  mxComplexDouble *array = mxGetComplexDoubles(arr);

  mxArray *pos = matGetVariable(pmat, "pos");
  double *posarr = NULL;
  if (pos != NULL) {
    posarr = mxGetPr(pos);
  }

  for (int i = 0; i < nelems; i++) {
    double famp = sqrt(array[i].real * array[i].real + array[i].imag * array[i].imag);
    uint8_t amp = static_cast<uint8_t>(std::clamp<double>(famp, 0, 1) * 255.99);
    double fphase = 0.0;
    if (amp != 0) fphase = atan2(array[i].imag, array[i].real);
    uint8_t phase = static_cast<uint8_t>(round((-fphase + M_PI) / (2.0 * M_PI) * 255.0));

    if (posarr != NULL) {
      double x = posarr[i * 3 + 0];
      double y = posarr[i * 3 + 1];
      double z = posarr[i * 3 + 2];
      Vector3 mtp = Vector3(x, y, z) * 10.0;
      Vector3 trp = this->geometry()->position(i);
      if ((mtp - trp).l2_norm() > 10) {
        std::cout << "Warning: position mismatch at " << i << std::endl << mtp << std::endl << trp << std::endl;
      }
    }

    this->_data[this->geometry()->deviceIdxForTransIdx(i)][i % NUM_TRANS_IN_UNIT] = ((uint16_t)amp << 8) + phase;
  }
#endif

  this->_built = true;
}
}  // namespace autd::gain
