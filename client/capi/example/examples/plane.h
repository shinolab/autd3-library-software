// File: plane.h
// Project: examples
// Created Date: 15/12/2021
// Author: Shun Suzuki
// -----
// Last Modified: 15/12/2021
// Modified By: Shun Suzuki (suzuki@hapis.k.u-tokyo.ac.jp)
// -----
// Copyright (c) 2021 Hapis Lab. All rights reserved.
//

#pragma once

void plane(void* autd) {
  AUTDSetSilentMode(autd, true);

  void* g = NULL;
  AUTDGainPlaneWave(&g, 0, 0, 1, 0xFF);

  void* m = NULL;
  AUTDModulationSine(&m, 150, 1.0, 0.5);

  AUTDSendHeaderBody(autd, m, g);

  AUTDDeleteGain(g);
  AUTDDeleteModulation(m);
}
