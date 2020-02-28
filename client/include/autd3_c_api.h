// File: autd3_c_api.h
// Project: include
// Created Date: 07/02/2018
// Author: Shun Suzuki
// -----
// Last Modified: 25/02/2020
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

#ifdef UNITY_DEBUG
using DebugLogFunc = void (*)(const char *);
#endif

#pragma region Controller
EXPORT void AUTDCreateController(AUTDControllerHandle *out);
EXPORT int AUTDOpenController(AUTDControllerHandle handle, int linkType, const char *location);
EXPORT int AUTDGetAdapterPointer(void **out);
EXPORT void AUTDGetAdapter(void *p_adapter, int index, char *descs, char *names);
EXPORT void AUTDFreeAdapterPointer(void *p_adapter);
EXPORT int AUTDAddDevice(AUTDControllerHandle handle, double x, double y, double z, double rz1, double ry, double rz2, int groupId);
EXPORT int AUTDAddDeviceQuaternion(AUTDControllerHandle handle, double x, double y, double z, double qua_w, double qua_x, double qua_y, double qua_z,
                                   int groupId);
EXPORT void AUTDDelDevice(AUTDControllerHandle handle, int devId);
EXPORT void AUTDCloseController(AUTDControllerHandle handle);
EXPORT void AUTDFreeController(AUTDControllerHandle handle);
EXPORT void AUTDSetSilentMode(AUTDControllerHandle handle, bool mode);
EXPORT void AUTDCalibrateModulation(AUTDControllerHandle handle);
EXPORT void AUTDStop(AUTDControllerHandle handle);
#pragma endregion

#pragma region Property
EXPORT bool AUTDIsOpen(AUTDControllerHandle handle);
EXPORT bool AUTDIsSilentMode(AUTDControllerHandle handle);
EXPORT int AUTDNumDevices(AUTDControllerHandle handle);
EXPORT int AUTDNumTransducers(AUTDControllerHandle handle);
EXPORT double AUTDFrequency(AUTDControllerHandle handle);
EXPORT size_t AUTDRemainingInBuffer(AUTDControllerHandle handle);
#pragma endregion

#pragma region Gain
EXPORT void AUTDFocalPointGain(AUTDGainPtr *gain, double x, double y, double z, uint8_t amp);
EXPORT void AUTDGroupedGain(AUTDGainPtr *gain, int *groupIDs, AUTDGainPtr *gains, int size);
EXPORT void AUTDBesselBeamGain(AUTDGainPtr *gain, double x, double y, double z, double n_x, double n_y, double n_z, double theta_z);
EXPORT void AUTDPlaneWaveGain(AUTDGainPtr *gain, double n_x, double n_y, double n_z);
EXPORT void AUTDCustomGain(AUTDGainPtr *gain, uint16_t *data, int dataLength);
EXPORT void AUTDHoloGain(AUTDGainPtr *gain, double *points, double *amps, int size);
EXPORT void AUTDTransducerTestGain(AUTDGainPtr *gain, int idx, int amp, int phase);
EXPORT void AUTDNullGain(AUTDGainPtr *gain);
EXPORT void AUTDDeleteGain(AUTDGainPtr gain);
#pragma endregion

#pragma region Modulation
EXPORT void AUTDModulation(AUTDModulationPtr *mod, uint8_t amp);
EXPORT void AUTDRawPCMModulation(AUTDModulationPtr *mod, const char *filename, double sampFreq);
EXPORT void AUTDSawModulation(AUTDModulationPtr *mod, int freq);
EXPORT void AUTDSineModulation(AUTDModulationPtr *mod, int freq, double amp, double offset);
EXPORT void AUTDDeleteModulation(AUTDModulationPtr mod);
#pragma endregion

#pragma region LowLevelInterface
EXPORT void AUTDAppendGain(AUTDControllerHandle handle, AUTDGainPtr gain);
EXPORT void AUTDAppendGainSync(AUTDControllerHandle handle, AUTDGainPtr gain);
EXPORT void AUTDAppendModulation(AUTDControllerHandle handle, AUTDModulationPtr mod);
EXPORT void AUTDAppendModulationSync(AUTDControllerHandle handle, AUTDModulationPtr mod);
EXPORT void AUTDAppendSTMGain(AUTDControllerHandle handle, AUTDGainPtr gain);
EXPORT void AUTDStartSTModulation(AUTDControllerHandle handle, double freq);
EXPORT void AUTDStopSTModulation(AUTDControllerHandle handle);
EXPORT void AUTDFinishSTModulation(AUTDControllerHandle handle);
EXPORT void AUTDSetGain(AUTDControllerHandle handle, int deviceIndex, int transIndex, int amp, int phase);
EXPORT void AUTDFlush(AUTDControllerHandle handle);
EXPORT int AUTDDevIdForDeviceIdx(AUTDControllerHandle handle, int devIdx);
EXPORT int AUTDDevIdForTransIdx(AUTDControllerHandle handle, int transIdx);
EXPORT double *AUTDTransPosition(AUTDControllerHandle handle, int transIdx);
EXPORT double *AUTDTransDirection(AUTDControllerHandle handle, int transIdx);
EXPORT double *GetAngleZYZ(double *rotationMatrix);
#pragma endregion

#pragma region Debug
#ifdef UNITY_DEBUG
EXPORT void DebugLog(const char *msg);
EXPORT void SetDebugLog(DebugLogFunc func);
EXPORT void DebugLogTest();
#endif
#pragma endregion
}
