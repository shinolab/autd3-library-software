/*
 * File: soem.c
 * Project: example
 * Created Date: 25/11/2021
 * Author: Shun Suzuki
 * -----
 * Last Modified: 14/12/2021
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
#include "soem_link.h"

void callback(char* msg) {
  printf_s("Link is lost\n");
  printf_s("%s\n", msg);
  exit(-1);
}

int main() {
  void* cnt = NULL;
  AUTDCreateController(&cnt);

  AUTDAddDevice(cnt, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
  // AUTDAddDeviceQuaternion(cnt, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);

  void* adapter_list = NULL;
  int32_t i;
  char name[128], desc[128];
  const int32_t adapter_list_size = AUTDGetAdapterPointer(&adapter_list);
  for (i = 0; i < adapter_list_size; i++) {
    AUTDGetAdapter(adapter_list, i, desc, name);
    printf_s("[%d]: %s, %s\n", i, desc, name);
  }
  printf_s("Choose number: ");
  (void)scanf_s("%d", &i);
  (void)getchar();
  AUTDGetAdapter(adapter_list, i, desc, name);
  void* link = NULL;
  const int32_t device_num = AUTDNumDevices(cnt);
  AUTDLinkSOEM(&link, name, device_num, 1);
  AUTDFreeAdapterPointer(adapter_list);

  AUTDSetSOEMOnLost(link, (void*)callback);

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
