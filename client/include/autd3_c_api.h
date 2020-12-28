// File: autd3_c_api.h
// Project: include
// Created Date: 07/02/2018
// Author: Shun Suzuki
// -----
// Last Modified: 27/12/2020
// Modified By: Shun Suzuki (suzuki@hapis.k.u-tokyo.ac.jp)
// -----
// Copyright (c) 2018-2020 Hapis Lab. All rights reserved.
//

#pragma once

#ifdef _DEBUG
#define UNITY_DEBUG
#endif

#if WIN32
#define EXPORT_AUTD_DLL __declspec(dllexport)
#else
#define EXPORT_AUTD_DLL __attribute__((visibility("default")))
#endif

extern "C" {

#ifdef UNITY_DEBUG
using DebugLogFunc = void (*)(const char*);
#endif

#pragma region Controller
EXPORT_AUTD_DLL void AUTDCreateController(void** out);
EXPORT_AUTD_DLL int32_t AUTDOpenControllerWith(void* handle, void* p_link);
EXPORT_AUTD_DLL int32_t AUTDAddDevice(void* handle, float x, float y, float z, float rz1, float ry, float rz2, int32_t group_id);
EXPORT_AUTD_DLL int32_t AUTDAddDeviceQuaternion(void* handle, float x, float y, float z, float qua_w, float qua_x, float qua_y, float qua_z,
                                                int32_t group_id);
EXPORT_AUTD_DLL bool AUTDCalibrate(void* handle, int32_t smpl_freq, int32_t buf_size);
EXPORT_AUTD_DLL void AUTDCloseController(void* handle);
EXPORT_AUTD_DLL void AUTDClear(void* handle);
EXPORT_AUTD_DLL void AUTDFreeController(void* handle);
EXPORT_AUTD_DLL void AUTDSetSilentMode(void* handle, bool mode);
EXPORT_AUTD_DLL void AUTDStop(void* handle);
EXPORT_AUTD_DLL int32_t AUTDGetAdapterPointer(void** out);
EXPORT_AUTD_DLL void AUTDGetAdapter(void* p_adapter, int32_t index, char* desc, char* name);
EXPORT_AUTD_DLL void AUTDFreeAdapterPointer(void* p_adapter);
EXPORT_AUTD_DLL int32_t AUTDGetFirmwareInfoListPointer(void* handle, void** out);
EXPORT_AUTD_DLL void AUTDGetFirmwareInfo(void* p_firm_info_list, int32_t index, char* cpu_ver, char* fpga_ver);
EXPORT_AUTD_DLL void AUTDFreeFirmwareInfoListPointer(void* p_firm_info_list);
#pragma endregion

#pragma region Property
EXPORT_AUTD_DLL bool AUTDIsOpen(void* handle);
EXPORT_AUTD_DLL bool AUTDIsSilentMode(void* handle);
EXPORT_AUTD_DLL float AUTDWavelength(void* handle);
EXPORT_AUTD_DLL void AUTDSetWavelength(void* handle, float wavelength);
EXPORT_AUTD_DLL void AUTDSetDelay(void* handle, uint16_t* delay, int32_t data_length);
EXPORT_AUTD_DLL int32_t AUTDNumDevices(void* handle);
EXPORT_AUTD_DLL int32_t AUTDNumTransducers(void* handle);
EXPORT_AUTD_DLL uint64_t AUTDRemainingInBuffer(void* handle);
#pragma endregion

#pragma region Gain
EXPORT_AUTD_DLL void AUTDFocalPointGain(void** gain, float x, float y, float z, uint8_t duty);
EXPORT_AUTD_DLL void AUTDGroupedGain(void** gain, const int32_t* group_ids, void** gains, int32_t size);
EXPORT_AUTD_DLL void AUTDBesselBeamGain(void** gain, float x, float y, float z, float n_x, float n_y, float n_z, float theta_z, uint8_t duty);
EXPORT_AUTD_DLL void AUTDPlaneWaveGain(void** gain, float n_x, float n_y, float n_z, uint8_t duty);
EXPORT_AUTD_DLL void AUTDCustomGain(void** gain, uint16_t* data, int32_t data_length);
EXPORT_AUTD_DLL void AUTDHoloGain(void** gain, float* points, float* amps, int32_t size, int32_t method, void* params);
EXPORT_AUTD_DLL void AUTDTransducerTestGain(void** gain, int32_t idx, uint8_t duty, uint8_t phase);
EXPORT_AUTD_DLL void AUTDNullGain(void** gain);
EXPORT_AUTD_DLL void AUTDDeleteGain(void* gain);
#pragma endregion

#pragma region Modulation
EXPORT_AUTD_DLL void AUTDModulation(void** mod, uint8_t amp);
EXPORT_AUTD_DLL void AUTDCustomModulation(void** mod, uint8_t* buf, uint32_t size);
EXPORT_AUTD_DLL void AUTDRawPCMModulation(void** mod, const char* filename, float sampling_freq);
EXPORT_AUTD_DLL void AUTDSawModulation(void** mod, int32_t freq);
EXPORT_AUTD_DLL void AUTDSineModulation(void** mod, int32_t freq, float amp, float offset);
EXPORT_AUTD_DLL void AUTDSquareModulation(void** mod, int32_t freq, uint8_t low, uint8_t high);
EXPORT_AUTD_DLL void AUTDWavModulation(void** mod, const char* filename);
EXPORT_AUTD_DLL void AUTDDeleteModulation(void* mod);
#pragma endregion

#pragma region Sequence
EXPORT_AUTD_DLL void AUTDSequence(void** out);
EXPORT_AUTD_DLL void AUTDSequenceAppendPoint(void* seq, float x, float y, float z);
EXPORT_AUTD_DLL void AUTDSequenceAppendPoints(void* seq, float* points, uint64_t size);
EXPORT_AUTD_DLL float AUTDSequenceSetFreq(void* seq, float freq);
EXPORT_AUTD_DLL float AUTDSequenceFreq(void* seq);
EXPORT_AUTD_DLL float AUTDSequenceSamplingFreq(void* seq);
EXPORT_AUTD_DLL uint16_t AUTDSequenceSamplingFreqDiv(void* seq);
EXPORT_AUTD_DLL void AUTDCircumSequence(void** out, float x, float y, float z, float nx, float ny, float nz, float radius, uint64_t n);
EXPORT_AUTD_DLL void AUTDDeleteSequence(void* seq);
#pragma endredion

#pragma region Link
EXPORT_AUTD_DLL void AUTDSOEMLink(void** out, const char* ifname, int32_t device_num);
EXPORT_AUTD_DLL void AUTDTwinCATLink(void** out, const char* ipv4_addr, const char* ams_net_id);
EXPORT_AUTD_DLL void AUTDLocalTwinCATLink(void** out);
EXPORT_AUTD_DLL void AUTDEmulatorLink(void** out, const char* addr, uint16_t port, void* handle);
#pragma endregion

#pragma region LowLevelInterface
EXPORT_AUTD_DLL void AUTDAppendGain(void* handle, void* gain);
EXPORT_AUTD_DLL void AUTDAppendGainSync(void* handle, void* gain, bool wait_for_send);
EXPORT_AUTD_DLL void AUTDAppendModulation(void* handle, void* mod);
EXPORT_AUTD_DLL void AUTDAppendModulationSync(void* handle, void* mod);
EXPORT_AUTD_DLL void AUTDAppendSTMGain(void* handle, void* gain);
EXPORT_AUTD_DLL void AUTDStartSTModulation(void* handle, float freq);
EXPORT_AUTD_DLL void AUTDStopSTModulation(void* handle);
EXPORT_AUTD_DLL void AUTDFinishSTModulation(void* handle);
EXPORT_AUTD_DLL void AUTDAppendSequence(void* handle, void* seq);
EXPORT_AUTD_DLL void AUTDFlush(void* handle);
EXPORT_AUTD_DLL int32_t AUTDDeviceIdxForTransIdx(void* handle, int32_t global_trans_idx);
EXPORT_AUTD_DLL float* AUTDTransPositionByGlobal(void* handle, int32_t global_trans_idx);
EXPORT_AUTD_DLL float* AUTDTransPositionByLocal(void* handle, int32_t device_idx, int32_t local_trans_idx);
EXPORT_AUTD_DLL float* AUTDDeviceDirection(void* handle, int32_t device_idx);
#pragma endregion

#pragma region Debug
#ifdef UNITY_DEBUG
EXPORT_AUTD_DLL void DebugLog(const char* msg);
EXPORT_AUTD_DLL void SetDebugLog(DebugLogFunc func);
#endif
#pragma endregion
}
