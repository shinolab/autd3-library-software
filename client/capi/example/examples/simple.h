// File: simple.h
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

#define TRANS_SPACING_MM (10.16)
#define NUM_TRANS_X (18)
#define NUM_TRANS_Y (14)

void simple(void* autd) {
  AUTDSetSilentMode(autd, true);

  double x = TRANS_SPACING_MM * ((NUM_TRANS_X - 1) / 2.0);
  double y = TRANS_SPACING_MM * ((NUM_TRANS_Y - 1) / 2.0);
  double z = 150.0;

  void* g = NULL;
  AUTDGainFocalPoint(&g, x, y, z, 0xFF);

  void* m = NULL;
  AUTDModulationSine(&m, 150, 1.0, 0.5);

  AUTDSendHeaderBody(autd, m, g);

  AUTDDeleteGain(g);
  AUTDDeleteModulation(m);
}
