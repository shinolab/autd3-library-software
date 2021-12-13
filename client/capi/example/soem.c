/*
 * File: soem.c
 * Project: example
 * Created Date: 25/11/2021
 * Author: Shun Suzuki
 * -----
 * Last Modified: 13/12/2021
 * Modified By: Shun Suzuki (suzuki@hapis.k.u-tokyo.ac.jp)
 * -----
 * Copyright (c) 2021 Hapis Lab. All rights reserved.
 *
 */

#define _AMD64_
#include <stdio.h>
#include <stdlib.h>
#include <windef.h>

#include "autd3_c_api.h"
#include "soem_link.h"

int main() {
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
  AUTDLinkSOEM(&link, name, 1, 1);
  AUTDFreeAdapterPointer(adapter_list);

  void* cnt = NULL;
  AUTDCreateController(&cnt);

  AUTDAddDevice(cnt, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
  if (!AUTDOpenController(cnt, link)) {
    const int32_t error_size = AUTDGetLastError(NULL);
    char* error = malloc(error_size);
    AUTDGetLastError(error);
    printf_s("%s\n", error);
    free(error);
    return -1;
  }

  AUTDClear(cnt);

  void* firm_info_list = NULL;
  const int32_t firm_info_list_size = AUTDGetFirmwareInfoListPointer(cnt, &firm_info_list);
  for (i = 0; i < firm_info_list_size; i++) {
    char cpu[128], fpga[128];
    AUTDGetFirmwareInfo(firm_info_list, i, cpu, fpga);
    printf_s("[%d]: CPU=%s, FPGA=%s\n", i, cpu, fpga);
  }
  AUTDFreeFirmwareInfoListPointer(firm_info_list);

  AUTDSetSilentMode(cnt, TRUE);

  void* g = NULL;
  AUTDGainFocalPoint(&g, 90.0, 70.0, 150.0, 0xFF);

  void* m = NULL;
  AUTDModulationSine(&m, 150, 1.0, 0.5);

  AUTDSendHeaderBody(cnt, g, m);

  printf_s("press any key to finish...\n");
  (void)getchar();

  AUTDCloseController(cnt);

  AUTDDeleteGain(g);
  AUTDDeleteModulation(m);
  AUTDFreeController(cnt);

  return 0;
}
