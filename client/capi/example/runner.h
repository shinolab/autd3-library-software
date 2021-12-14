// File: runner.h
// Project: example
// Created Date: 14/12/2021
// Author: Shun Suzuki
// -----
// Last Modified: 14/12/2021
// Modified By: Shun Suzuki (suzuki@hapis.k.u-tokyo.ac.jp)
// -----
// Copyright (c) 2021 Hapis Lab. All rights reserved.
//

#pragma once

#include <stdlib.h>

#include "autd3_c_api.h"
#include "examples/bessel.h"
#include "examples/group.h"
#include "examples/seq.h"
#include "examples/seq_gain.h"
#include "examples/simple.h"
#include "examples/stm.h"
#ifdef BUILD_HOLO_GAIN
#include "examples/holo.h"
#endif
#ifdef BUILD_FROM_FILE_MOD
#include "examples/mod_from_file.h"
#endif

typedef void (*TestFunction)(void*);

typedef struct {
  const char* name;
  TestFunction func;
} Test;

int run(void* autd) {
  int32_t example_size = 4;
#ifdef BUILD_HOLO_GAIN
  example_size++;
#endif
#ifdef BUILD_FROM_FILE_MOD
  example_size++;
#endif
  if (AUTDNumDevices(autd) == 2) example_size++;

  Test* examples = (Test*)malloc(example_size * sizeof(Test));

  int idx = 0;
  examples[idx].name = "Simple";
  examples[idx++].func = simple;
  examples[idx].name = "Bessel";
  examples[idx++].func = bessel;
#ifdef BUILD_HOLO_GAIN
  examples[idx].name = "Holo";
  examples[idx++].func = holo;
#endif
#ifdef BUILD_FROM_FILE_MOD
  examples[idx].name = "Modulation from file";
  examples[idx++].func = mod_from_file;
#endif
  examples[idx].name = "STM";
  examples[idx++].func = stm;
  examples[idx].name = "PointSequence";
  examples[idx++].func = seq;
  examples[idx].name = "GainSequence";
  examples[idx++].func = seq_gain;
  if (AUTDNumDevices(autd) == 2) {
    examples[idx].name = "Grouped";
    examples[idx++].func = group;
  }

  AUTDSetWavelength(autd, 8.5);
  const double wavelength = AUTDGetWavelength(autd);
  printf_s("Wavelength: %lf mm\n", wavelength);

  AUTDClear(autd);

  void* firm_info_list = NULL;
  const int32_t firm_info_list_size = AUTDGetFirmwareInfoListPointer(autd, &firm_info_list);
  for (int32_t i = 0; i < firm_info_list_size; i++) {
    char cpu[128], fpga[128];
    AUTDGetFirmwareInfo(firm_info_list, i, cpu, fpga);
    printf_s("[%d]: CPU=%s, FPGA=%s\n", i, cpu, fpga);
  }
  AUTDFreeFirmwareInfoListPointer(firm_info_list);

  while (1) {
    for (int32_t i = 0; i < example_size; i++) {
      printf_s("[%d]: %s\n", i, examples[i].name);
    }
    printf_s("[Others]: finish.\n");

    printf_s("Choose number: ");
    int32_t i;
    if (!scanf_s("%d", &i)) {
      return 0;
    }
    (void)getchar();
    if (i >= example_size) {
      return 0;
    }

    examples[i].func(autd);

    printf_s("press any key to finish...");
    (void)getchar();

    printf_s("Finish.\n");
    AUTDStop(autd);
  }

  AUTDClear(autd);
  AUTDCloseController(autd);

  AUTDFreeController(autd);

  free(examples);

  return 0;
}
