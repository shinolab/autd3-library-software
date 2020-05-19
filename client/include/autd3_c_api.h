// File: autd3_c_api.h
// Project: include
// Created Date: 07/02/2018
// Author: Shun Suzuki
// -----
// Last Modified: 19/05/2020
// Modified By: Shun Suzuki (suzuki@hapis.k.u-tokyo.ac.jp)
// -----
// Copyright (c) 2018-2020 Hapis Lab. All rights reserved.
//

#pragma once

#ifdef _DEBUG
#define UNITY_DEBUG
#endif

#if WIN32
#define EXPORT __declspec(dllexport)
#else
#define EXPORT __attribute__((visibility("default")))
#endif

extern "C" {
using AUTDControllerHandle = void *;
using AUTDGainPtr = void *;
using AUTDModulationPtr = void *;
using AUTDLinkPtr = void *;

#ifdef UNITY_DEBUG
using DebugLogFunc = void (*)(const char *);
#endif

#pragma region Controller
EXPORT void AUTDCreateController(AUTDControllerHandle *out);
EXPORT int32_t AUTDOpenController(AUTDControllerHandle handle, int32_t linkType, const char *location);
EXPORT int32_t AUTDOpenControllerWith(AUTDControllerHandle handle, AUTDLinkPtr link);
EXPORT int32_t AUTDAddDevice(AUTDControllerHandle handle, double x, double y, double z, double rz1, double ry, double rz2, int32_t groupId);
EXPORT int32_t AUTDAddDeviceQuaternion(AUTDControllerHandle handle, double x, double y, double z, double qua_w, double qua_x, double qua_y,
                                       double qua_z, int32_t groupId);
EXPORT void AUTDDelDevice(AUTDControllerHandle handle, int32_t devId);
EXPORT void AUTDCloseController(AUTDControllerHandle handle);
EXPORT void AUTDFreeController(AUTDControllerHandle handle);
EXPORT void AUTDSetSilentMode(AUTDControllerHandle handle, bool mode);
EXPORT bool AUTDCalibrateModulation(AUTDControllerHandle handle);
EXPORT void AUTDStop(AUTDControllerHandle handle);
EXPORT int32_t AUTDGetAdapterPointer(void **out);
EXPORT void AUTDGetAdapter(void *p_adapter, int32_t index, char *descs, char *names);
EXPORT void AUTDFreeAdapterPointer(void *p_adapter);
EXPORT int32_t AUTDGetFirmwareInfoListPointer(AUTDControllerHandle handle, void **out);
EXPORT void AUTDGetFirmwareInfo(void *pfirminfolist, int32_t index, char *cpu_ver, char *fpga_ver);
EXPORT void AUTDFreeFirmwareInfoListPointer(void *pfirminfolist);
#pragma endregion

#pragma region Property
EXPORT bool AUTDIsOpen(AUTDControllerHandle handle);
EXPORT bool AUTDIsSilentMode(AUTDControllerHandle handle);
EXPORT int32_t AUTDNumDevices(AUTDControllerHandle handle);
EXPORT int32_t AUTDNumTransducers(AUTDControllerHandle handle);
EXPORT uint64_t AUTDRemainingInBuffer(AUTDControllerHandle handle);
#pragma endregion

#pragma region Gain
EXPORT void AUTDFocalPointGain(AUTDGainPtr *gain, double x, double y, double z, uint8_t amp);
EXPORT void AUTDGroupedGain(AUTDGainPtr *gain, int32_t *groupIDs, AUTDGainPtr *gains, int32_t size);
EXPORT void AUTDBesselBeamGain(AUTDGainPtr *gain, double x, double y, double z, double n_x, double n_y, double n_z, double theta_z);
EXPORT void AUTDPlaneWaveGain(AUTDGainPtr *gain, double n_x, double n_y, double n_z);
EXPORT void AUTDCustomGain(AUTDGainPtr *gain, uint16_t *data, int32_t dataLength);
EXPORT void AUTDHoloGain(AUTDGainPtr *gain, double *points, double *amps, int32_t size);
EXPORT void AUTDTransducerTestGain(AUTDGainPtr *gain, int32_t idx, int32_t amp, int32_t phase);
EXPORT void AUTDNullGain(AUTDGainPtr *gain);
EXPORT void AUTDDeleteGain(AUTDGainPtr gain);
#pragma endregion

#pragma region Modulation
EXPORT void AUTDModulation(AUTDModulationPtr *mod, uint8_t amp);
EXPORT void AUTDRawPCMModulation(AUTDModulationPtr *mod, const char *filename, double sampFreq);
EXPORT void AUTDSawModulation(AUTDModulationPtr *mod, int32_t freq);
EXPORT void AUTDSineModulation(AUTDModulationPtr *mod, int32_t freq, double amp, double offset);
EXPORT void AUTDWavModulation(AUTDModulationPtr *mod, const char *filename);
EXPORT void AUTDDeleteModulation(AUTDModulationPtr mod);
#pragma endregion

#pragma region Link
EXPORT void AUTDSOEMLink(AUTDLinkPtr *out, const char *ifname, int32_t device_num);
EXPORT void AUTDEtherCATLink(AUTDLinkPtr *out, const char *ipv4addr, const char *ams_net_id);
EXPORT void AUTDLocalEtherCATLink(AUTDLinkPtr *out);
EXPORT void AUTDEmulatorLink(AUTDLinkPtr *out, const char *addr, int32_t port, AUTDControllerHandle handle);
#pragma endregion

#pragma region LowLevelInterface
EXPORT void AUTDAppendGain(AUTDControllerHandle handle, AUTDGainPtr gain);
EXPORT void AUTDAppendGainSync(AUTDControllerHandle handle, AUTDGainPtr gain, bool wait_for_send);
EXPORT void AUTDAppendModulation(AUTDControllerHandle handle, AUTDModulationPtr mod);
EXPORT void AUTDAppendModulationSync(AUTDControllerHandle handle, AUTDModulationPtr mod);
EXPORT void AUTDAppendSTMGain(AUTDControllerHandle handle, AUTDGainPtr gain);
EXPORT void AUTDStartSTModulation(AUTDControllerHandle handle, double freq);
EXPORT void AUTDStopSTModulation(AUTDControllerHandle handle);
EXPORT void AUTDFinishSTModulation(AUTDControllerHandle handle);
EXPORT void AUTDFlush(AUTDControllerHandle handle);
EXPORT int32_t AUTDDevIdForDeviceIdx(AUTDControllerHandle handle, int32_t devIdx);
EXPORT int32_t AUTDDevIdForTransIdx(AUTDControllerHandle handle, int32_t transIdx);
EXPORT double *AUTDTransPosition(AUTDControllerHandle handle, int32_t transIdx);
EXPORT double *AUTDTransDirection(AUTDControllerHandle handle, int32_t transIdx);
#pragma endregion

#pragma region Debug
#ifdef UNITY_DEBUG
EXPORT void DebugLog(const char *msg);
EXPORT void SetDebugLog(DebugLogFunc func);
#endif
#pragma endregion
}
