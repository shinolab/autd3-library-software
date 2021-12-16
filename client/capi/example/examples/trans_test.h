// File: trans_test.h
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

#include <string.h>

#define NUM_TRANS_IN_UNIT (249)

void trans_test(void* autd) {
  AUTDSetSilentMode(autd, false);

  const int32_t num_trans = AUTDNumDevices(autd) * NUM_TRANS_IN_UNIT;

  uint8_t* duty_offsets = malloc(num_trans);
  (duty_offsets, 0, num_trans);
  duty_offsets[0] = 1;
  AUTDSetDelayOffset(autd, NULL, duty_offsets);

  void* g = NULL;
  AUTDGainTransducerTest(&g, 0, 0xFF, 0);

  void* m = NULL;
  AUTDModulationStatic(&m, 0xFF);

  AUTDSendHeaderBody(autd, m, g);

  AUTDDeleteGain(g);
  AUTDDeleteModulation(m);
  free(duty_offsets);
}
