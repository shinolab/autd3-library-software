// File: mod_from_file.h
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

#include "from_file_modulation.h"

#define TRANS_SPACING_MM (10.16)
#define NUM_TRANS_X (18)
#define NUM_TRANS_Y (14)

void mod_from_file(void* autd) {
  AUTDSetSilentMode(autd, true);

  double x = TRANS_SPACING_MM * ((NUM_TRANS_X - 1) / 2.0);
  double y = TRANS_SPACING_MM * ((NUM_TRANS_Y - 1) / 2.0);
  double z = 150.0;

  void* g = NULL;
  AUTDGainFocalPoint(&g, x, y, z, 0xFF);

  char path[256];
  sprintf_s(path, 256, "%s\\%s", AUTD3_RESOURCE_PATH, "sin150.wav");

  void* m = NULL;
  AUTDModulationWav(&m, path, 10);

  AUTDSendHeaderBody(autd, m, g);

  AUTDDeleteGain(g);
  AUTDDeleteModulation(m);
}
