/*
 * File: soem.c
 * Project: example
 * Created Date: 25/11/2021
 * Author: Shun Suzuki
 * -----
 * Last Modified: 25/11/2021
 * Modified By: Shun Suzuki (suzuki@hapis.k.u-tokyo.ac.jp)
 * -----
 * Copyright (c) 2021 Hapis Lab. All rights reserved.
 *
 */

#define _AMD64_
#include <libloaderapi.h>
#include <stdint.h>
#include <stdio.h>
#include <windef.h>

typedef void (*AUTDCreateController)(void** out);
typedef void (*AUTDFreeController)(void*);
typedef BOOL (*AUTDOpenController)(const void* handle, void* p_link);
typedef int32_t (*AUTDAddDevice)(const void* handle, double x, double y, double z, double rz1, double ry, double rz2);
typedef int32_t (*AUTDCloseController)(const void* handle);
typedef int32_t (*AUTDClear)(const void* handle);
typedef int32_t (*AUTDGetFirmwareInfoListPointer)(const void* handle, void** out);
typedef void (*AUTDGetFirmwareInfo)(const void* p_firm_info_list, int32_t index, char* cpu_ver, char* fpga_ver);
typedef void (*AUTDFreeFirmwareInfoListPointer)(const void* p_firm_info_list);
typedef void (*AUTDSetSilentMode)(const void* handle, BOOL mode);
typedef void (*AUTDGainFocalPoint)(void** gain, double x, double y, double z, uint8_t duty);
typedef void (*AUTDDeleteGain)(const void* gain);
typedef void (*AUTDModulationSine)(void** mod, int32_t freq, double amp, double offset);
typedef void (*AUTDDeleteModulation)(const void* mod);
typedef int32_t (*AUTDSendGainModulation)(const void* handle, const void* gain, const void* mod);

typedef int32_t (*AUTDGetAdapterPointer)(void** out);
typedef void (*AUTDGetAdapter)(void* p_adapter, int32_t index, char* desc, char* name);
typedef void (*AUTDFreeAdapterPointer)(void* p_adapter);
typedef void (*AUTDLinkSOEM)(void** out, const char* ifname, int32_t device_num, uint32_t cycle_ticks);

int main() {
  const HMODULE dll_base = LoadLibrary("autd3capi.dll");
  const HMODULE dll_soem = LoadLibrary("autd3capi-soem-link.dll");

  if (dll_base == NULL || dll_soem == NULL) {
    printf("failed to load dlls\n");
    return -1;
  }

  const AUTDCreateController create_controller = (AUTDCreateController)GetProcAddress(dll_base, "AUTDCreateController");
  const AUTDFreeController free_controller = (AUTDFreeController)GetProcAddress(dll_base, "AUTDFreeController");
  const AUTDOpenController open_controller = (AUTDOpenController)GetProcAddress(dll_base, "AUTDOpenController");
  const AUTDAddDevice add_device = (AUTDAddDevice)GetProcAddress(dll_base, "AUTDAddDevice");
  const AUTDCloseController close_controller = (AUTDCloseController)GetProcAddress(dll_base, "AUTDCloseController");
  const AUTDClear clear = (AUTDClear)GetProcAddress(dll_base, "AUTDClear");
  const AUTDGetFirmwareInfoListPointer get_firmware_info_list_pointer =
      (AUTDGetFirmwareInfoListPointer)GetProcAddress(dll_base, "AUTDGetFirmwareInfoListPointer");
  const AUTDGetFirmwareInfo get_firmware_info = (AUTDGetFirmwareInfo)GetProcAddress(dll_base, "AUTDGetFirmwareInfo");
  const AUTDFreeFirmwareInfoListPointer free_firmware_info_list_pointer =
      (AUTDFreeFirmwareInfoListPointer)GetProcAddress(dll_base, "AUTDFreeFirmwareInfoListPointer");
  const AUTDSetSilentMode set_silent_mode = (AUTDSetSilentMode)GetProcAddress(dll_base, "AUTDSetSilentMode");
  const AUTDGainFocalPoint gain_focal_point = (AUTDGainFocalPoint)GetProcAddress(dll_base, "AUTDGainFocalPoint");
  const AUTDDeleteGain delete_gain = (AUTDDeleteGain)GetProcAddress(dll_base, "AUTDDeleteGain");
  const AUTDModulationSine modulation_sine = (AUTDModulationSine)GetProcAddress(dll_base, "AUTDModulationSine");
  const AUTDDeleteModulation delete_modulation = (AUTDDeleteModulation)GetProcAddress(dll_base, "AUTDDeleteModulation");
  const AUTDSendGainModulation send_gain_modulation = (AUTDSendGainModulation)GetProcAddress(dll_base, "AUTDSendGainModulation");

  const AUTDGetAdapterPointer get_adapter_pointer = (AUTDGetAdapterPointer)GetProcAddress(dll_soem, "AUTDGetAdapterPointer");
  const AUTDGetAdapter get_adapter = (AUTDGetAdapter)GetProcAddress(dll_soem, "AUTDGetAdapter");
  const AUTDFreeAdapterPointer free_adapter_pointer = (AUTDFreeAdapterPointer)GetProcAddress(dll_soem, "AUTDFreeAdapterPointer");
  const AUTDLinkSOEM link_soem = (AUTDLinkSOEM)GetProcAddress(dll_soem, "AUTDLinkSOEM");

  void* adapter_list = NULL;
  int32_t i;
  char name[128], desc[128];
  const int32_t adapter_list_size = get_adapter_pointer(&adapter_list);
  for (i = 0; i < adapter_list_size; i++) {
    get_adapter(adapter_list, i, desc, name);
    printf("[%d]: %s, %s\n", i, desc, name);
  }
  printf("Choose number: ");
  (void)scanf_s("%d", &i);
  (void)getchar();
  get_adapter(adapter_list, i, desc, name);
  void* link = NULL;
  link_soem(&link, name, 1, 1);
  free_adapter_pointer(adapter_list);

  void* cnt = NULL;
  create_controller(&cnt);

  add_device(cnt, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
  open_controller(cnt, link);

  clear(cnt);

  void* firm_info_list = NULL;
  const int32_t firm_info_list_size = get_firmware_info_list_pointer(cnt, &firm_info_list);
  for (i = 0; i < firm_info_list_size; i++) {
    char cpu[128], fpga[128];
    get_firmware_info(firm_info_list, i, cpu, fpga);
    printf("[%d]: CPU=%s, FPGA=%s\n", i, cpu, fpga);
  }
  free_firmware_info_list_pointer(firm_info_list);

  set_silent_mode(cnt, TRUE);

  void* g = NULL;
  gain_focal_point(&g, 90.0, 70.0, 150.0, 0xFF);

  void* m = NULL;
  modulation_sine(&m, 150, 1.0, 0.5);

  send_gain_modulation(cnt, g, m);

  printf("press any key to finish...\n");
  (void)getchar();

  close_controller(cnt);

  delete_gain(g);
  delete_modulation(m);
  free_controller(cnt);

  FreeLibrary(dll_base);
  FreeLibrary(dll_soem);

  return 0;
}
