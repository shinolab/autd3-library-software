/*
 * File: twincat.c
 * Project: example
 * Created Date: 15/12/2021
 * Author: Shun Suzuki
 * -----
 * Last Modified: 15/12/2021
 * Modified By: Shun Suzuki (suzuki@hapis.k.u-tokyo.ac.jp)
 * -----
 * Copyright (c) 2021 Hapis Lab. All rights reserved.
 *
 */

#define _AMD64_
#include <stdio.h>
#include <stdlib.h>

#include "autd3_c_api.h"
#include "runner.h"
#include "twincat_link.h"

int main() {
  void* cnt = NULL;
  AUTDCreateController(&cnt);

  AUTDAddDevice(cnt, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
  // AUTDAddDeviceQuaternion(cnt, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);

  void* link = NULL;
  AUTDLinkTwinCAT(&link);

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
