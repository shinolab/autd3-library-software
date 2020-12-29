// File: autd3_c_api.h
// Project: include
// Created Date: 07/02/2018
// Author: Shun Suzuki
// -----
// Last Modified: 28/12/2020
// Modified By: Shun Suzuki (suzuki@hapis.k.u-tokyo.ac.jp)
// -----
// Copyright (c) 2018-2020 Hapis Lab. All rights reserved.
//

#pragma once

#if WIN32
#define EXPORT_AUTD_DLL __declspec(dllexport)
#else
#define EXPORT_AUTD_DLL __attribute__((visibility("default")))
#endif

#define VOID_PTR void*

extern "C" {

#pragma region Controller
EXPORT_AUTD_DLL void AUTDCreateController(VOID_PTR* out);
EXPORT_AUTD_DLL int32_t AUTDOpenControllerWith(VOID_PTR handle, VOID_PTR p_link);
EXPORT_AUTD_DLL int32_t AUTDAddDevice(VOID_PTR handle, float x, float y, float z, float rz1, float ry, float rz2, int32_t group_id);
EXPORT_AUTD_DLL int32_t AUTDAddDeviceQuaternion(VOID_PTR handle, float x, float y, float z, float qua_w, float qua_x, float qua_y, float qua_z,
                                                int32_t group_id);
EXPORT_AUTD_DLL bool AUTDCalibrate(VOID_PTR handle, int32_t smpl_freq, int32_t buf_size);
EXPORT_AUTD_DLL void AUTDCloseController(VOID_PTR handle);
EXPORT_AUTD_DLL void AUTDClear(VOID_PTR handle);
EXPORT_AUTD_DLL void AUTDFreeController(VOID_PTR handle);
EXPORT_AUTD_DLL void AUTDSetSilentMode(VOID_PTR handle, bool mode);
EXPORT_AUTD_DLL void AUTDStop(VOID_PTR handle);
EXPORT_AUTD_DLL int32_t AUTDGetAdapterPointer(VOID_PTR* out);
EXPORT_AUTD_DLL void AUTDGetAdapter(VOID_PTR p_adapter, int32_t index, char* desc, char* name);
EXPORT_AUTD_DLL void AUTDFreeAdapterPointer(VOID_PTR p_adapter);
EXPORT_AUTD_DLL int32_t AUTDGetFirmwareInfoListPointer(VOID_PTR handle, VOID_PTR* out);
EXPORT_AUTD_DLL void AUTDGetFirmwareInfo(VOID_PTR p_firm_info_list, int32_t index, char* cpu_ver, char* fpga_ver);
EXPORT_AUTD_DLL void AUTDFreeFirmwareInfoListPointer(VOID_PTR p_firm_info_list);
#pragma endregion

#pragma region Property
EXPORT_AUTD_DLL bool AUTDIsOpen(VOID_PTR handle);
EXPORT_AUTD_DLL bool AUTDIsSilentMode(VOID_PTR handle);
EXPORT_AUTD_DLL float AUTDWavelength(VOID_PTR handle);
EXPORT_AUTD_DLL void AUTDSetWavelength(VOID_PTR handle, float wavelength);
EXPORT_AUTD_DLL void AUTDSetDelay(VOID_PTR handle, const uint16_t* delay, int32_t data_length);
EXPORT_AUTD_DLL int32_t AUTDNumDevices(VOID_PTR handle);
EXPORT_AUTD_DLL int32_t AUTDNumTransducers(VOID_PTR handle);
EXPORT_AUTD_DLL uint64_t AUTDRemainingInBuffer(VOID_PTR handle);
#pragma endregion

#pragma region Gain
EXPORT_AUTD_DLL void AUTDFocalPointGain(VOID_PTR* gain, float x, float y, float z, uint8_t duty);
EXPORT_AUTD_DLL void AUTDGroupedGain(VOID_PTR* gain, const int32_t* group_ids, VOID_PTR const* in_gains, int32_t size);
EXPORT_AUTD_DLL void AUTDBesselBeamGain(VOID_PTR* gain, float x, float y, float z, float n_x, float n_y, float n_z, float theta_z, uint8_t duty);
EXPORT_AUTD_DLL void AUTDPlaneWaveGain(VOID_PTR* gain, float n_x, float n_y, float n_z, uint8_t duty);
EXPORT_AUTD_DLL void AUTDCustomGain(VOID_PTR* gain, const uint16_t* data, int32_t data_length);
EXPORT_AUTD_DLL void AUTDHoloGain(VOID_PTR* gain, const float* points, const float* amps, int32_t size, int32_t method, VOID_PTR params);
EXPORT_AUTD_DLL void AUTDTransducerTestGain(VOID_PTR* gain, int32_t idx, uint8_t duty, uint8_t phase);
EXPORT_AUTD_DLL void AUTDNullGain(VOID_PTR* gain);
EXPORT_AUTD_DLL void AUTDDeleteGain(VOID_PTR gain);
#pragma endregion

#pragma region Modulation
EXPORT_AUTD_DLL void AUTDModulation(VOID_PTR* mod, uint8_t amp);
EXPORT_AUTD_DLL void AUTDCustomModulation(VOID_PTR* mod, const uint8_t* buf, uint32_t size);
EXPORT_AUTD_DLL void AUTDRawPCMModulation(VOID_PTR* mod, const char* filename, float sampling_freq);
EXPORT_AUTD_DLL void AUTDSawModulation(VOID_PTR* mod, int32_t freq);
EXPORT_AUTD_DLL void AUTDSineModulation(VOID_PTR* mod, int32_t freq, float amp, float offset);
EXPORT_AUTD_DLL void AUTDSquareModulation(VOID_PTR* mod, int32_t freq, uint8_t low, uint8_t high);
EXPORT_AUTD_DLL void AUTDWavModulation(VOID_PTR* mod, const char* filename);
EXPORT_AUTD_DLL void AUTDDeleteModulation(VOID_PTR mod);
#pragma endregion

#pragma region Sequence
EXPORT_AUTD_DLL void AUTDSequence(VOID_PTR* out);
EXPORT_AUTD_DLL void AUTDSequenceAppendPoint(VOID_PTR seq, float x, float y, float z);
EXPORT_AUTD_DLL void AUTDSequenceAppendPoints(VOID_PTR seq, const float* points, uint64_t size);
EXPORT_AUTD_DLL float AUTDSequenceSetFreq(VOID_PTR seq, float freq);
EXPORT_AUTD_DLL float AUTDSequenceFreq(VOID_PTR seq);
EXPORT_AUTD_DLL float AUTDSequenceSamplingFreq(VOID_PTR seq);
EXPORT_AUTD_DLL uint16_t AUTDSequenceSamplingFreqDiv(VOID_PTR seq);
EXPORT_AUTD_DLL void AUTDCircumSequence(VOID_PTR* out, float x, float y, float z, float nx, float ny, float nz, float radius, uint64_t n);
EXPORT_AUTD_DLL void AUTDDeleteSequence(VOID_PTR seq);
#pragma endredion

#pragma region Link
EXPORT_AUTD_DLL void AUTDSOEMLink(VOID_PTR* out, const char* ifname, int32_t device_num);
EXPORT_AUTD_DLL void AUTDTwinCATLink(VOID_PTR* out, const char* ipv4_addr, const char* ams_net_id);
EXPORT_AUTD_DLL void AUTDLocalTwinCATLink(VOID_PTR* out);
EXPORT_AUTD_DLL void AUTDEmulatorLink(VOID_PTR* out, const char* addr, uint16_t port, VOID_PTR handle);
#pragma endregion

#pragma region LowLevelInterface
EXPORT_AUTD_DLL void AUTDAppendGain(VOID_PTR handle, VOID_PTR gain);
EXPORT_AUTD_DLL void AUTDAppendGainSync(VOID_PTR handle, VOID_PTR gain, bool wait_for_send);
EXPORT_AUTD_DLL void AUTDAppendModulation(VOID_PTR handle, VOID_PTR mod);
EXPORT_AUTD_DLL void AUTDAppendModulationSync(VOID_PTR handle, VOID_PTR mod);
EXPORT_AUTD_DLL void AUTDAppendSTMGain(VOID_PTR handle, VOID_PTR gain);
EXPORT_AUTD_DLL void AUTDStartSTModulation(VOID_PTR handle, float freq);
EXPORT_AUTD_DLL void AUTDStopSTModulation(VOID_PTR handle);
EXPORT_AUTD_DLL void AUTDFinishSTModulation(VOID_PTR handle);
EXPORT_AUTD_DLL void AUTDAppendSequence(VOID_PTR handle, VOID_PTR seq);
EXPORT_AUTD_DLL void AUTDFlush(VOID_PTR handle);
EXPORT_AUTD_DLL int32_t AUTDDeviceIdxForTransIdx(VOID_PTR handle, int32_t global_trans_idx);
EXPORT_AUTD_DLL float* AUTDTransPositionByGlobal(VOID_PTR handle, int32_t global_trans_idx);
EXPORT_AUTD_DLL float* AUTDTransPositionByLocal(VOID_PTR handle, int32_t device_idx, int32_t local_trans_idx);
EXPORT_AUTD_DLL float* AUTDDeviceDirection(VOID_PTR handle, int32_t device_idx);
#pragma endregion
}
