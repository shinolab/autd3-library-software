// File: autd3_c_api.h
// Project: include
// Created Date: 07/02/2018
// Author: Shun Suzuki
// -----
// Last Modified: 25/12/2020
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

#ifdef UNITY_DEBUG
using DebugLogFunc = void (*)(const char*);
#endif

#pragma region Controller
EXPORT void AUTDCreateController(void** out);
EXPORT int32_t AUTDOpenControllerWith(void* handle, void* p_link);
EXPORT int32_t AUTDAddDevice(void* handle, double x, double y, double z, double rz1, double ry, double rz2, int32_t group_id);
EXPORT int32_t AUTDAddDeviceQuaternion(void* handle, double x, double y, double z, double qua_w, double qua_x, double qua_y, double qua_z,
                                       int32_t group_id);
EXPORT bool AUTDCalibrate(void* handle, int32_t smpl_freq, int32_t buf_size);
EXPORT void AUTDCloseController(void* handle);
EXPORT void AUTDClear(void* handle);
EXPORT void AUTDFreeController(void* handle);
EXPORT void AUTDSetSilentMode(void* handle, bool mode);
EXPORT void AUTDStop(void* handle);
EXPORT int32_t AUTDGetAdapterPointer(void** out);
EXPORT void AUTDGetAdapter(void* p_adapter, int32_t index, char* desc, char* name);
EXPORT void AUTDFreeAdapterPointer(void* p_adapter);
EXPORT int32_t AUTDGetFirmwareInfoListPointer(void* handle, void** out);
EXPORT void AUTDGetFirmwareInfo(void* p_firm_info_list, int32_t index, char* cpu_ver, char* fpga_ver);
EXPORT void AUTDFreeFirmwareInfoListPointer(void* p_firm_info_list);
#pragma endregion

#pragma region Property
EXPORT bool AUTDIsOpen(void* handle);
EXPORT bool AUTDIsSilentMode(void* handle);
EXPORT int32_t AUTDNumDevices(void* handle);
EXPORT int32_t AUTDNumTransducers(void* handle);
EXPORT uint64_t AUTDRemainingInBuffer(void* handle);
#pragma endregion

#pragma region Gain
EXPORT void AUTDFocalPointGain(void** gain, double x, double y, double z, uint8_t duty);
EXPORT void AUTDGroupedGain(void** gain, const int32_t* group_ids, void** gains, int32_t size);
EXPORT void AUTDBesselBeamGain(void** gain, double x, double y, double z, double n_x, double n_y, double n_z, double theta_z, uint8_t duty);
EXPORT void AUTDPlaneWaveGain(void** gain, double n_x, double n_y, double n_z, uint8_t duty);
EXPORT void AUTDCustomGain(void** gain, uint16_t* data, int32_t dataLength);
EXPORT void AUTDHoloGain(void** gain, double* points, double* amps, int32_t size, int32_t method, void* params);
EXPORT void AUTDTransducerTestGain(void** gain, int32_t idx, uint8_t duty, uint8_t phase);
EXPORT void AUTDNullGain(void** gain);
EXPORT void AUTDDeleteGain(void* gain);
#pragma endregion

#pragma region Modulation
EXPORT void AUTDModulation(void** mod, uint8_t amp);
EXPORT void AUTDCustomModulation(void** mod, uint8_t* buf, uint32_t size);
EXPORT void AUTDRawPCMModulation(void** mod, const char* filename, double sampling_freq);
EXPORT void AUTDSawModulation(void** mod, int32_t freq);
EXPORT void AUTDSineModulation(void** mod, int32_t freq, double amp, double offset);
EXPORT void AUTDSquareModulation(void** mod, int32_t freq, uint8_t low, uint8_t high);
EXPORT void AUTDWavModulation(void** mod, const char* filename);
EXPORT void AUTDDeleteModulation(void* mod);
#pragma endregion

#pragma region Sequence
EXPORT void AUTDSequence(void** out);
EXPORT void AUTDSequenceAppendPoint(void* seq, double x, double y, double z);
EXPORT void AUTDSequenceAppendPoints(void* seq, double* points, uint64_t size);
EXPORT double AUTDSequenceSetFreq(void* seq, double freq);
EXPORT double AUTDSequenceFreq(void* seq);
EXPORT double AUTDSequenceSamplingFreq(void* seq);
EXPORT uint16_t AUTDSequenceSamplingFreqDiv(void* seq);
EXPORT void AUTDCircumSequence(void** out, double x, double y, double z, double nx, double ny, double nz, double radius, uint64_t n);
EXPORT void AUTDDeleteSequence(void* seq);
#pragma endredion

#pragma region Link
EXPORT void AUTDSOEMLink(void** out, const char* ifname, int32_t device_num);
EXPORT void AUTDTwinCATLink(void** out, const char* ipv4addr, const char* ams_net_id);
EXPORT void AUTDLocalTwinCATLink(void** out);
EXPORT void AUTDEmulatorLink(void** out, const char* addr, const uint16_t port, void* handle);
#pragma endregion

#pragma region LowLevelInterface
EXPORT void AUTDAppendGain(void* handle, void* gain);
EXPORT void AUTDAppendGainSync(void* handle, void* gain, bool wait_for_send);
EXPORT void AUTDAppendModulation(void* handle, void* mod);
EXPORT void AUTDAppendModulationSync(void* handle, void* mod);
EXPORT void AUTDAppendSTMGain(void* handle, void* gain);
EXPORT void AUTDStartSTModulation(void* handle, double freq);
EXPORT void AUTDStopSTModulation(void* handle);
EXPORT void AUTDFinishSTModulation(void* handle);
EXPORT void AUTDAppendSequence(void* handle, void* seq);
EXPORT void AUTDFlush(void* handle);
EXPORT int32_t AUTDDevIdxForTransIdx(void* handle, int32_t trans_idx);
EXPORT double* AUTDTransPosition(void* handle, int32_t trans_idx);
EXPORT double* AUTDTransDirection(void* handle, int32_t trans_idx);
#pragma endregion

#pragma region Debug
#ifdef UNITY_DEBUG
EXPORT void DebugLog(const char* msg);
EXPORT void SetDebugLog(DebugLogFunc func);
#endif
#pragma endregion
}
