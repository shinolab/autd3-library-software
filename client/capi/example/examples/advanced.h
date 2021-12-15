// File: advanced.h
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

#define NUM_TRANS_IN_UNIT (249)

void advanced(void* autd) {
  AUTDSetSilentMode(autd, false);

  const int32_t num_trans = AUTDNumDevices(autd) * NUM_TRANS_IN_UNIT;
  uint8_t* delays = malloc(num_trans);
  memset(delays, 0, num_trans);
  delays[0] = 4;
  AUTDSetDelayOffset(autd, delays, NULL);

  uint16_t* uniform_gain = malloc(num_trans * sizeof(uint16_t));
  for (int32_t i = 0; i < num_trans; i++) {
    uniform_gain[i] = 0xFF80;
  }
  void* g = NULL;
  AUTDGainCustom(&g, uniform_gain, num_trans);

  uint8_t* burst_modulation = malloc(4000);
  memset(burst_modulation, 0, 4000);
  burst_modulation[0] = 0xFF;
  void* m = NULL;
  AUTDModulationCustom(&m, burst_modulation, 4000, 10);

  AUTDSendHeaderBody(autd, m, g);

  AUTDDeleteGain(g);
  AUTDDeleteModulation(m);
  free(delays);
  free(uniform_gain);
  free(burst_modulation);
}
