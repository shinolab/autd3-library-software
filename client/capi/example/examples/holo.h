// File: holo.h
// Project: examples
// Created Date: 14/12/2021
// Author: Shun Suzuki
// -----
// Last Modified: 14/12/2021
// Modified By: Shun Suzuki (suzuki@hapis.k.u-tokyo.ac.jp)
// -----
// Copyright (c) 2021 Hapis Lab. All rights reserved.
//

#pragma once

#include "holo_gain.h"

#define TRANS_SPACING_MM (10.16)
#define NUM_TRANS_X (18)
#define NUM_TRANS_Y (14)

typedef struct {
  const char* name;
  void* gain;
} Opt;

void* select_opt(void* backend, double* foci, double* amps) {
  printf_s("Select Optimization Method (default is SDP)\n");

  int32_t opt_size = 10;
  Opt* opts = (Opt*)malloc(opt_size * sizeof(Opt));

  int idx = 0;
  opts[idx].name = "SDP";
  AUTDGainHoloSDP(&opts[idx++].gain, backend, foci, amps, 2, 1e-3, 0.9, 100, true);
  opts[idx].name = "EVD";
  AUTDGainHoloEVD(&opts[idx++].gain, backend, foci, amps, 2, 1.0, true);
  opts[idx].name = "GS";
  AUTDGainHoloGS(&opts[idx++].gain, backend, foci, amps, 2, 100);
  opts[idx].name = "GSPAT";
  AUTDGainHoloGSPAT(&opts[idx++].gain, backend, foci, amps, 2, 100);
  opts[idx].name = "NAIVE";
  AUTDGainHoloNaive(&opts[idx++].gain, backend, foci, amps, 2);
  opts[idx].name = "LM";
  AUTDGainHoloLM(&opts[idx++].gain, backend, foci, amps, 2, 1e-8, 1e-8, 1e-3, 5, NULL, 0);
  opts[idx].name = "GaussNewton";
  AUTDGainHoloGaussNewton(&opts[idx++].gain, backend, foci, amps, 2, 1e-6, 1e-6, 500, NULL, 0);
  opts[idx].name = "GradientDescent";
  AUTDGainHoloGradientDescent(&opts[idx++].gain, backend, foci, amps, 2, 1e-6, 0.5, 2000, NULL, 0);
  opts[idx].name = "APO";
  AUTDGainHoloAPO(&opts[idx++].gain, backend, foci, amps, 2, 1e-8, 1.0, 200);
  opts[idx].name = "Greedy";
  AUTDGainHoloGreedy(&opts[idx++].gain, backend, foci, amps, 2, 16);
  for (int32_t i = 0; i < opt_size; i++) {
    printf_s("[%d]: %s\n", i, opts[i].name);
  }

  idx = 0;
  if (!scanf_s("%d", &idx)) {
    idx = 0;
  }
  (void)getchar();
  if (idx >= opt_size) {
    idx = 0;
  }

  for (int32_t i = 0; i < opt_size; i++) {
    if (i == idx) continue;
    AUTDDeleteGain(opts[i].gain);
  }

  return opts[idx].gain;
}

void holo(void* autd) {
  AUTDSetSilentMode(autd, true);

  void* m = NULL;
  AUTDModulationSine(&m, 150, 1.0, 0.5);

  double x = TRANS_SPACING_MM * ((NUM_TRANS_X - 1) / 2.0);
  double y = TRANS_SPACING_MM * ((NUM_TRANS_Y - 1) / 2.0);
  double z = 150.0;

  double foci[6] = {x - 30.0, y, z, x + 30.0, y, z};
  double amps[2] = {1.0, 1.0};

  void* backend = NULL;
  AUTDEigen3Backend(&backend);

  void* g = select_opt(backend, foci, amps);
  AUTDSendHeaderBody(autd, m, g);

  AUTDDeleteGain(g);
  AUTDDeleteModulation(m);
  AUTDDeleteBackend(backend);
}
