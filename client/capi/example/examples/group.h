// File: group.h
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

void group(void* autd) {
  AUTDSetSilentMode(autd, true);

  double x = TRANS_SPACING_MM * (((double)NUM_TRANS_X - 1.0) / 2.0);
  double y = TRANS_SPACING_MM * (((double)NUM_TRANS_Y - 1.0) / 2.0);

  void* g1 = NULL;
  AUTDGainFocalPoint(&g1, x, y, 150.0, 0xFF);

  void* g2 = NULL;
  AUTDGainBesselBeam(&g2, x, y, 0.0, 0.0, 0.0, 1.0, 13.0 / 180.0 * M_PI, 0xFF);

  void* g = NULL;
  AUTDGainGrouped(&g, autd);

  AUTDGainGroupedAdd(g, 0, g1);
  AUTDGainGroupedAdd(g, 1, g2);

  void* m = NULL;
  AUTDModulationSine(&m, 150, 1.0, 0.5);

  AUTDSendHeaderBody(autd, m, g);

  AUTDDeleteGain(g);
  AUTDDeleteModulation(m);
}
