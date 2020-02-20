// File: matlabgain.cpp
// Project: lib
// Created Date: 20/09/2016
// Author:Seki Inoue
// -----
// Last Modified: 20/02/2020
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

#include <complex>
#include <iostream>

#include "autd3.hpp"
#include "controller.hpp"
#include "gain.hpp"
#include "privdef.hpp"

template <typename T>
T clamp(T val, T _min, T _max) {
  return std::max<T>(std::min<T>(val, _max), _min);
}

autd::GainPtr autd::MatlabGain::Create(std::string filename, std::string varname) {
#ifndef MATLAB_ENABLED
  throw new std::runtime_error(
      "MatlabGain requires Matlab libraries. Recompile with Matlab "
      "Environment.");
#endif
  auto ptr = CreateHelper<MatlabGain>();
  ptr->_filename = filename;
  ptr->_varname = varname;
  return ptr;
}

void autd::MatlabGain::Build() {
  if (this->built()) return;
  if (this->geometry() == nullptr) throw new std::runtime_error("Geometry is required to build Gain");

#ifdef MATLAB_ENABLED
  this->_data.clear();
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

  this->_data.clear();
  const int ndevice = this->geometry()->numDevices();
  for (int i = 0; i < ndevice; i++) {
    this->_data[this->geometry()->deviceIdForDeviceIdx(i)].resize(NUM_TRANS_IN_UNIT);
  }
  for (int i = 0; i < nelems; i++) {
    float famp = sqrtf(array[i].real * array[i].real + array[i].imag * array[i].imag);
    uint8_t amp = clamp<float>(famp, 0, 1) * 255.99;
    float fphase = 0.0f;
    if (amp != 0) fphase = atan2f(array[i].imag, array[i].real);
    uint8_t phase = round((-fphase + M_PI) / (2.0 * M_PI) * 255.0);

    if (posarr != NULL) {
      double x = posarr[i * 3 + 0];
      double y = posarr[i * 3 + 1];
      double z = posarr[i * 3 + 2];
      Eigen::Vector3f mtp = Eigen::Vector3f(x, y, z) * 10.0;
      Eigen::Vector3f trp = this->geometry()->position(i);
      if ((mtp - trp).norm() > 10) {
        std::cout << "Warning: position mismatch at " << i << std::endl << mtp << std::endl << trp << std::endl;
      }
    }

    this->_data[this->geometry()->deviceIdForTransIdx(i)][i % NUM_TRANS_IN_UNIT] = ((uint16_t)amp << 8) + phase;
  }
#endif

  this->_built = true;
}
