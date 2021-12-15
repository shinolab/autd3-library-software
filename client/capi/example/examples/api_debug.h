// File: api_debug.h
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

void api_debug(void* autd) {
  void* g = NULL;
  AUTDGainFocalPoint(&g, 90, 70, 150, 0xFF);
  void* m = NULL;
  printf_s("SineSquared\n");
  AUTDModulationSineSquared(&m, 150, 1.0, 0.5);

  AUTDSendHeaderBody(autd, m, g);
  AUTDDeleteGain(g);
  AUTDDeleteModulation(m);

  printf_s("press enter to test SineLegacy...\n");
  (void)getchar();

  AUTDModulationSineLegacy(&m, 150.0, 1.0, 0.5);
  AUTDSendHeader(autd, m);
  AUTDDeleteModulation(m);

  printf_s("press enter to test Square...\n");
  (void)getchar();

  AUTDModulationSquare(&m, 150, 0x00, 0xFF, 0.5);
  AUTDSendHeader(autd, m);

  AUTDModulationSetSamplingFreqDiv(m, 5);

  printf_s("Modulation API Test\n");
  printf_s("Modulation sampling frequency division: %ld\n", AUTDModulationSamplingFreqDiv(m));
  printf_s("Modulation sampling frequency: %lf Hz\n", AUTDModulationSamplingFreq(m));
  AUTDDeleteModulation(m);

  printf_s("Sequence API Test\n");
  void* seq = NULL;
  AUTDSequence(&seq);
  const int32_t point_num = 200;
  for (int32_t i = 0; i < point_num; i++) AUTDSequenceAddPoint(seq, 0.0, 0.0, 0.0, 0xFF);
  AUTDSequenceSetFreq(seq, 1.0);
  printf_s("Actual frequency is %lf Hz\n", AUTDSequenceFreq(seq));
  printf_s("Actual period is %ld us\n", AUTDSequencePeriod(seq));
  printf_s("Sampling frequency is %lf us\n", AUTDSequenceSamplingFreq(seq));
  printf_s("Sampling period is %ld us\n", AUTDSequenceSamplingPeriod(seq));
  printf_s("Sampling frequency division is %ld\n", AUTDSequenceSamplingFreqDiv(seq));

  printf_s("Set sampling frequency division to 100\n");
  AUTDSequenceSetSamplingFreqDiv(seq, 100);
  printf_s("Actual frequency is %lf Hz\n", AUTDSequenceFreq(seq));
  printf_s("Actual period is %ld us\n", AUTDSequencePeriod(seq));
  printf_s("Sampling frequency is %lf us\n", AUTDSequenceSamplingFreq(seq));
  printf_s("Sampling period is %ld us\n", AUTDSequenceSamplingPeriod(seq));
  printf_s("Sampling frequency division is %ld\n", AUTDSequenceSamplingFreqDiv(seq));
  AUTDDeleteSequence(seq);

  printf_s("press enter to pause...\n");
  (void)getchar();

  AUTDPause(autd);

  printf_s("press enter to resume...\n");
  (void)getchar();
  AUTDResume(autd);

  printf_s("press enter to stop...\n");
  (void)getchar();

  AUTDGainNull(&g);
  AUTDSendBody(autd, g);

  AUTDDeleteGain(g);
}
