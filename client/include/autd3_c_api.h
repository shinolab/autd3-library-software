// File: autd3_c_api.h
// Project: include
// Created Date: 07/02/2018
// Author: Shun Suzuki
// -----
// Last Modified: 21/02/2020
// Modified By: Shun Suzuki (suzuki@hapis.k.u-tokyo.ac.jp)
// -----
// Copyright (c) 2018-2020 Hapis Lab. All rights reserved.
//

#ifndef INCLUDE_AUTD3_C_API_H_
#define INCLUDE_AUTD3_C_API_H_

#ifdef _DEBUG
#define UNITY_DEBUG
#endif

extern "C" {
using AUTDControllerHandle = void *;
using AUTDGainPtr = void *;
using AUTDModulationPtr = void *;

#ifdef UNITY_DEBUG
using DebugLogFunc = void (*)(const char *);
#endif

#pragma region Controller
__declspec(dllexport) void AUTDCreateController(AUTDControllerHandle *out);
__declspec(dllexport) int AUTDOpenController(AUTDControllerHandle handle, int linkType, const char *location);
__declspec(dllexport) int AUTDGetAdapterPointer(void **out);
__declspec(dllexport) void AUTDGetAdapter(void *p_adapter, int index, char *descs, char *names);
__declspec(dllexport) void AUTDFreeAdapterPointer(void *p_adapter);
__declspec(dllexport) int AUTDAddDevice(AUTDControllerHandle handle, double x, double y, double z, double rz1, double ry, double rz2, int groupId);
__declspec(dllexport) int AUTDAddDeviceQuaternion(AUTDControllerHandle handle, double x, double y, double z, double qua_w, double qua_x, double qua_y,
                                                  double qua_z, int groupId);
__declspec(dllexport) void AUTDDelDevice(AUTDControllerHandle handle, int devId);
__declspec(dllexport) void AUTDCloseController(AUTDControllerHandle handle);
__declspec(dllexport) void AUTDFreeController(AUTDControllerHandle handle);
__declspec(dllexport) void AUTDSetSilentMode(AUTDControllerHandle handle, bool mode);
__declspec(dllexport) void AUTDCalibrateModulation(AUTDControllerHandle handle);
__declspec(dllexport) void AUTDStop(AUTDControllerHandle handle);
#pragma endregion

#pragma region Property
__declspec(dllexport) bool AUTDIsOpen(AUTDControllerHandle handle);
__declspec(dllexport) bool AUTDIsSilentMode(AUTDControllerHandle handle);
__declspec(dllexport) int AUTDNumDevices(AUTDControllerHandle handle);
__declspec(dllexport) int AUTDNumTransducers(AUTDControllerHandle handle);
__declspec(dllexport) double AUTDFrequency(AUTDControllerHandle handle);
__declspec(dllexport) size_t AUTDRemainingInBuffer(AUTDControllerHandle handle);
#pragma endregion

#pragma region Gain
__declspec(dllexport) void AUTDFocalPointGain(AUTDGainPtr *gain, double x, double y, double z, uint8_t amp);
__declspec(dllexport) void AUTDGroupedGain(AUTDGainPtr *gain, int *groupIDs, AUTDGainPtr *gains, int size);
__declspec(dllexport) void AUTDBesselBeamGain(AUTDGainPtr *gain, double x, double y, double z, double n_x, double n_y, double n_z, double theta_z);
__declspec(dllexport) void AUTDPlaneWaveGain(AUTDGainPtr *gain, double n_x, double n_y, double n_z);
__declspec(dllexport) void AUTDCustomGain(AUTDGainPtr *gain, uint16_t *data, int dataLength);
__declspec(dllexport) void AUTDHoloGain(AUTDGainPtr *gain, double *points, double *amps, int size);
__declspec(dllexport) void AUTDTransducerTestGain(AUTDGainPtr *gain, int idx, int amp, int phase);
__declspec(dllexport) void AUTDNullGain(AUTDGainPtr *gain);
__declspec(dllexport) void AUTDDeleteGain(AUTDGainPtr gain);
#pragma endregion

#pragma region Modulation
__declspec(dllexport) void AUTDModulation(AUTDModulationPtr *mod, uint8_t amp);
__declspec(dllexport) void AUTDRawPCMModulation(AUTDModulationPtr *mod, const char *filename, double sampFreq);
__declspec(dllexport) void AUTDSawModulation(AUTDModulationPtr *mod, int freq);
__declspec(dllexport) void AUTDSineModulation(AUTDModulationPtr *mod, int freq, double amp, double offset);
__declspec(dllexport) void AUTDDeleteModulation(AUTDModulationPtr mod);
#pragma endregion

#pragma region LowLevelInterface
__declspec(dllexport) void AUTDAppendGain(AUTDControllerHandle handle, AUTDGainPtr gain);
__declspec(dllexport) void AUTDAppendGainSync(AUTDControllerHandle handle, AUTDGainPtr gain);
__declspec(dllexport) void AUTDAppendModulation(AUTDControllerHandle handle, AUTDModulationPtr mod);
__declspec(dllexport) void AUTDAppendModulationSync(AUTDControllerHandle handle, AUTDModulationPtr mod);
__declspec(dllexport) void AUTDAppendSTMGain(AUTDControllerHandle handle, AUTDGainPtr gain);
__declspec(dllexport) void AUTDStartSTModulation(AUTDControllerHandle handle, double freq);
__declspec(dllexport) void AUTDStopSTModulation(AUTDControllerHandle handle);
__declspec(dllexport) void AUTDFinishSTModulation(AUTDControllerHandle handle);
__declspec(dllexport) void AUTDSetGain(AUTDControllerHandle handle, int deviceIndex, int transIndex, int amp, int phase);
__declspec(dllexport) void AUTDFlush(AUTDControllerHandle handle);
__declspec(dllexport) int AUTDDevIdForDeviceIdx(AUTDControllerHandle handle, int devIdx);
__declspec(dllexport) int AUTDDevIdForTransIdx(AUTDControllerHandle handle, int transIdx);
__declspec(dllexport) double *AUTDTransPosition(AUTDControllerHandle handle, int transIdx);
__declspec(dllexport) double *AUTDTransDirection(AUTDControllerHandle handle, int transIdx);
__declspec(dllexport) double *GetAngleZYZ(double *rotationMatrix);

__declspec(dllexport) void AUTDAppendLateralGain(AUTDControllerHandle handle, AUTDGainPtr gain);
__declspec(dllexport) void AUTDStartLateralModulation(AUTDControllerHandle handle, double freq);
__declspec(dllexport) void AUTDFinishLateralModulation(AUTDControllerHandle handle);
__declspec(dllexport) void AUTDResetLateralGain(AUTDControllerHandle handle);
#pragma endregion

#pragma region Debug
#ifdef UNITY_DEBUG
__declspec(dllexport) void DebugLog(const char *msg);
__declspec(dllexport) void SetDebugLog(DebugLogFunc func);
__declspec(dllexport) void DebugLogTest();
#endif
#pragma endregion
}
#endif  // INCLUDE_AUTD3_C_API_H_
