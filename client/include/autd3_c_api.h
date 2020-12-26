// File: autd3_c_api.h
// Project: include
// Created Date: 07/02/2018
// Author: Shun Suzuki
// -----
// Last Modified: 26/12/2020
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
EXPORT int32_t AUTDAddDevice(void* handle, float x, float y, float z, float rz1, float ry, float rz2, int32_t group_id);
EXPORT int32_t AUTDAddDeviceQuaternion(void* handle, float x, float y, float z, float qua_w, float qua_x, float qua_y, float qua_z, int32_t group_id);
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
EXPORT float AUTDWavelength(void* handle);
EXPORT void AUTDSetWavelength(void* handle, float wavelength);
EXPORT void AUTDSetDelay(void* handle, uint16_t* delay, int32_t data_length);
EXPORT int32_t AUTDNumDevices(void* handle);
EXPORT int32_t AUTDNumTransducers(void* handle);
EXPORT uint64_t AUTDRemainingInBuffer(void* handle);
#pragma endregion

#pragma region Gain
EXPORT void AUTDFocalPointGain(void** gain, float x, float y, float z, uint8_t duty);
EXPORT void AUTDGroupedGain(void** gain, const int32_t* group_ids, void** gains, int32_t size);
EXPORT void AUTDBesselBeamGain(void** gain, float x, float y, float z, float n_x, float n_y, float n_z, float theta_z, uint8_t duty);
EXPORT void AUTDPlaneWaveGain(void** gain, float n_x, float n_y, float n_z, uint8_t duty);
EXPORT void AUTDCustomGain(void** gain, uint16_t* data, int32_t data_length);
EXPORT void AUTDHoloGain(void** gain, float* points, float* amps, int32_t size, int32_t method, void* params);
EXPORT void AUTDTransducerTestGain(void** gain, int32_t idx, uint8_t duty, uint8_t phase);
EXPORT void AUTDNullGain(void** gain);
EXPORT void AUTDDeleteGain(void* gain);
#pragma endregion

#pragma region Modulation
EXPORT void AUTDModulation(void** mod, uint8_t amp);
EXPORT void AUTDCustomModulation(void** mod, uint8_t* buf, uint32_t size);
EXPORT void AUTDRawPCMModulation(void** mod, const char* filename, float sampling_freq);
EXPORT void AUTDSawModulation(void** mod, int32_t freq);
EXPORT void AUTDSineModulation(void** mod, int32_t freq, float amp, float offset);
EXPORT void AUTDSquareModulation(void** mod, int32_t freq, uint8_t low, uint8_t high);
EXPORT void AUTDWavModulation(void** mod, const char* filename);
EXPORT void AUTDDeleteModulation(void* mod);
#pragma endregion

#pragma region Sequence
EXPORT void AUTDSequence(void** out);
EXPORT void AUTDSequenceAppendPoint(void* seq, float x, float y, float z);
EXPORT void AUTDSequenceAppendPoints(void* seq, float* points, uint64_t size);
EXPORT float AUTDSequenceSetFreq(void* seq, float freq);
EXPORT float AUTDSequenceFreq(void* seq);
EXPORT float AUTDSequenceSamplingFreq(void* seq);
EXPORT uint16_t AUTDSequenceSamplingFreqDiv(void* seq);
EXPORT void AUTDCircumSequence(void** out, float x, float y, float z, float nx, float ny, float nz, float radius, uint64_t n);
EXPORT void AUTDDeleteSequence(void* seq);
#pragma endredion

#pragma region Link
EXPORT void AUTDSOEMLink(void** out, const char* ifname, int32_t device_num);
EXPORT void AUTDTwinCATLink(void** out, const char* ipv4_addr, const char* ams_net_id);
EXPORT void AUTDLocalTwinCATLink(void** out);
EXPORT void AUTDEmulatorLink(void** out, const char* addr, uint16_t port, void* handle);
#pragma endregion

#pragma region LowLevelInterface
EXPORT void AUTDAppendGain(void* handle, void* gain);
EXPORT void AUTDAppendGainSync(void* handle, void* gain, bool wait_for_send);
EXPORT void AUTDAppendModulation(void* handle, void* mod);
EXPORT void AUTDAppendModulationSync(void* handle, void* mod);
EXPORT void AUTDAppendSTMGain(void* handle, void* gain);
EXPORT void AUTDStartSTModulation(void* handle, float freq);
EXPORT void AUTDStopSTModulation(void* handle);
EXPORT void AUTDFinishSTModulation(void* handle);
EXPORT void AUTDAppendSequence(void* handle, void* seq);
EXPORT void AUTDFlush(void* handle);
EXPORT int32_t AUTDDeviceIdxForTransIdx(void* handle, int32_t global_trans_idx);
EXPORT float* AUTDTransPositionByGlobal(void* handle, int32_t global_trans_idx);
EXPORT float* AUTDTransPositionByLocal(void* handle, int32_t device_idx, int32_t local_trans_idx);
EXPORT float* AUTDDeviceDirection(void* handle, int32_t device_idx);
#pragma endregion

#pragma region Debug
#ifdef UNITY_DEBUG
EXPORT void DebugLog(const char* msg);
EXPORT void SetDebugLog(DebugLogFunc func);
#endif
#pragma endregion
}
