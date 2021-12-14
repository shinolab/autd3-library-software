// File: seq_gain.h
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

#include <math.h>

#define TRANS_SPACING_MM (10.16)
#define NUM_TRANS_X (18)
#define NUM_TRANS_Y (14)

#define DUTY_PHASE_FULL (0x0001)
#define PHASE_FULL (0x0002)
#define PHASE_HALF (0x0004)

void seq_gain(void* autd) {
  AUTDSetSilentMode(autd, false);

  double x = TRANS_SPACING_MM * ((NUM_TRANS_X - 1) / 2.0);
  double y = TRANS_SPACING_MM * ((NUM_TRANS_Y - 1) / 2.0);
  double z = 150.0;

  void* seq = NULL;
  AUTDGainSequence(&seq, autd, DUTY_PHASE_FULL);

  const int32_t point_num = 200;
  for (int32_t i = 0; i < point_num; i++) {
    const double radius = 30.0;
    const double theta = 2.0 * M_PI * (double)i / (double)point_num;
    void* g = NULL;
    AUTDGainFocalPoint(&g, x + radius * cos(theta), y + radius * sin(theta), z, 0xFF);
    AUTDSequenceAddGain(seq, g);
    AUTDDeleteGain(g);
  }

  const uint32_t v = AUTDSequenceSamplingFreqDiv(seq);
  printf_s("Sample frequency division is %d\n", v);

  const double actual_freq = AUTDSequenceSetFreq(seq, 1.0);
  printf_s("Actual frequency is %lf Hz\n", actual_freq);

  void* m = NULL;
  AUTDModulationStatic(&m, 0xFF);

  AUTDSendHeaderBody(autd, m, seq);

  AUTDDeleteSequence(seq);
  AUTDDeleteModulation(m);
}
