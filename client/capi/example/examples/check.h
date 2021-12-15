// File: check.h
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

void check(void* autd) {
  int32_t num_devices = AUTDNumDevices(autd);
  printf_s("===== Device informations =====\n");
  for (int32_t i = 0; i < num_devices; i++) {
    double x, y, z;
    AUTDTransPosition(autd, i, 0, &x, &y, &z);
    printf_s("[%d]: Origin = (%lf, %lf, %lf)\n", i, x, y, z);
    AUTDDeviceXDirection(autd, i, &x, &y, &z);
    printf_s("[%d]: X = (%lf, %lf, %lf)\n", i, x, y, z);
    AUTDDeviceYDirection(autd, i, &x, &y, &z);
    printf_s("[%d]: Y = (%lf, %lf, %lf)\n", i, x, y, z);
    AUTDDeviceZDirection(autd, i, &x, &y, &z);
    printf_s("[%d]: Z = (%lf, %lf, %lf)\n", i, x, y, z);
  }
  printf_s("\n");

  printf_s("===== Flags =====\n");

  AUTDSetOutputEnable(autd, false);
  AUTDSetSilentMode(autd, true);
  AUTDSetReadsFPGAInfo(autd, true);
  AUTDSetOutputBalance(autd, false);
  AUTDSetCheckAck(autd, false);
  AUTDSetForceFan(autd, false);

  bool is_enable = AUTDGetOutputEnable(autd);
  bool is_silent = AUTDGetSilentMode(autd);
  bool is_force_fan = AUTDGetForceFan(autd);
  bool is_reads_fpga_info = AUTDGetReadsFPGAInfo(autd);
  bool is_balance = AUTDGetOutputBalance(autd);
  bool is_check_ack = AUTDGetCheckAck(autd);

  printf_s("Is enable: %d\n", is_enable);
  printf_s("Is silent: %d\n", is_silent);
  printf_s("Is force fan: %d\n", is_force_fan);
  printf_s("Is reads FPGA info: %d\n", is_reads_fpga_info);
  printf_s("Is balance: %d\n", is_balance);
  printf_s("Is check ack: %d\n", is_check_ack);
  printf_s("\n");

  printf_s("===== Properties =====\n");

  AUTDSetWavelength(autd, 8.5);
  AUTDSetAttenuation(autd, 0.0);

  printf_s("Wavelength %lf mm\n", AUTDGetWavelength(autd));
  printf_s("Attenuation coefficient %lf [Np/mm]\n", AUTDGetAttenuation(autd));
  printf_s("\n");

  printf_s("===== FPGA informations =====\n");

  uint8_t* infos = malloc(num_devices);
  AUTDGetFPGAInfo(autd, infos);
  for (int32_t i = 0; i < num_devices; i++) {
    printf_s("[%d]: Is fan running : %d\n", i, infos[i]);
  }
  printf_s("\n");

  printf_s("press any key to force fan...");
  (void)getchar();

  AUTDSetForceFan(autd, true);
  AUTDUpdateCtrlFlags(autd);

  Sleep(100);

  AUTDGetFPGAInfo(autd, infos);
  for (int32_t i = 0; i < num_devices; i++) {
    printf_s("[%d]: Is fan running : %d\n", i, infos[i]);
  }
  printf_s("\n");

  printf_s("press any key to stop fan...");
  (void)getchar();

  AUTDSetForceFan(autd, false);
  AUTDUpdateCtrlFlags(autd);

  free(infos);
}
