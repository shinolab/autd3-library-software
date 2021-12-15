// File: stm.h
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

void stm(void* autd) {
  AUTDSetSilentMode(autd, true);

  void* m = NULL;
  AUTDModulationStatic(&m, 0xFF);

  AUTDSendHeader(autd, m);

  double x = TRANS_SPACING_MM * (((double)NUM_TRANS_X - 1.0) / 2.0);
  double y = TRANS_SPACING_MM * (((double)NUM_TRANS_Y - 1.0) / 2.0);
  double z = 150.0;

  void* stm = NULL;
  AUTDSTMController(&stm, autd);

  const int32_t point_num = 200;
  for (int32_t i = 0; i < point_num; i++) {
    const double radius = 30.0;
    const double theta = 2.0 * M_PI * (double)i / (double)point_num;
    void* g = NULL;
    AUTDGainFocalPoint(&g, x + radius * cos(theta), y + radius * sin(theta), z, 0xFF);
    AUTDAddSTMGain(stm, g);
    AUTDDeleteGain(g);
  }

  AUTDStartSTM(stm, 0.5);

  printf_s("press any key to stop...");
  (void)getchar();

  AUTDStopSTM(stm);
  AUTDFinishSTM(stm);

  AUTDDeleteModulation(m);
}
