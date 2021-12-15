/*
 * File: emulator.c
 * Project: example
 * Created Date: 14/12/2021
 * Author: Shun Suzuki
 * -----
 * Last Modified: 15/12/2021
 * Modified By: Shun Suzuki (suzuki@hapis.k.u-tokyo.ac.jp)
 * -----
 * Copyright (c) 2021 Hapis Lab. All rights reserved.
 *
 */

#include <stdio.h>
#include <stdlib.h>

#include "autd3_c_api.h"
#include "emulator_link.h"
#include "runner.h"

int main() {
  void* cnt = NULL;
  AUTDCreateController(&cnt);

  AUTDAddDevice(cnt, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
  // AUTDAddDeviceQuaternion(cnt, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);

  void* link = NULL;
  AUTDLinkEmulator(&link, 50632, cnt);

  if (!AUTDOpenController(cnt, link)) {
    const int32_t error_size = AUTDGetLastError(NULL);
    char* error = malloc(error_size);
    AUTDGetLastError(error);
    printf_s("%s\n", error);
    free(error);
    return -1;
  }

  return run(cnt);
}