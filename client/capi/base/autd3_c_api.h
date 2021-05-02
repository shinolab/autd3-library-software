// File: autd3_c_api.h
// Project: include
// Created Date: 07/02/2018
// Author: Shun Suzuki
// -----
// Last Modified: 02/05/2021
// Modified By: Shun Suzuki (suzuki@hapis.k.u-tokyo.ac.jp)
// -----
// Copyright (c) 2018-2020 Hapis Lab. All rights reserved.
//

#pragma once

#if WIN32
#define EXPORT_AUTD __declspec(dllexport)
#else
#define EXPORT_AUTD __attribute__((visibility("default")))
#endif

extern "C" {

#pragma region Controller
EXPORT_AUTD void AUTDCreateController(void** out);
EXPORT_AUTD bool AUTDOpenControllerWith(void* handle, void* p_link);
EXPORT_AUTD int32_t AUTDAddDevice(void* handle, float x, float y, float z, float rz1, float ry, float rz2, int32_t group_id);
EXPORT_AUTD int32_t AUTDAddDeviceQuaternion(void* handle, float x, float y, float z, float qua_w, float qua_x, float qua_y, float qua_z,
                                            int32_t group_id);
EXPORT_AUTD int32_t AUTDDeleteDevice(void* handle, int32_t idx);
EXPORT_AUTD void AUTDClearDevices(void* handle);
EXPORT_AUTD bool AUTDSynchronize(void* handle, int32_t smpl_freq, int32_t buf_size);
EXPORT_AUTD bool AUTDCloseController(void* handle);
EXPORT_AUTD bool AUTDClear(void* handle);
EXPORT_AUTD void AUTDFreeController(void* handle);
EXPORT_AUTD void AUTDSetSilentMode(void* handle, bool mode);
EXPORT_AUTD bool AUTDStop(void* handle);
EXPORT_AUTD int32_t AUTDGetFirmwareInfoListPointer(void* handle, void** out);
EXPORT_AUTD void AUTDGetFirmwareInfo(void* p_firm_info_list, int32_t index, char* cpu_ver, char* fpga_ver);
EXPORT_AUTD void AUTDFreeFirmwareInfoListPointer(void* p_firm_info_list);
EXPORT_AUTD int32_t AUTDGetLastError(char* error);
#pragma endregion

#pragma region Property
EXPORT_AUTD bool AUTDIsOpen(void* handle);
EXPORT_AUTD bool AUTDIsSilentMode(void* handle);
EXPORT_AUTD float AUTDWavelength(void* handle);
EXPORT_AUTD void AUTDSetWavelength(void* handle, float wavelength);
EXPORT_AUTD int32_t AUTDNumDevices(void* handle);
EXPORT_AUTD int32_t AUTDNumTransducers(void* handle);
EXPORT_AUTD uint64_t AUTDRemainingInBuffer(void* handle);
#pragma endregion

#pragma region Gain
EXPORT_AUTD void AUTDNullGain(void** gain);
EXPORT_AUTD void AUTDGroupedGain(void** gain, int32_t* group_ids, void** in_gains, int32_t size);
EXPORT_AUTD void AUTDDeleteGain(void* gain);
EXPORT_AUTD void AUTDFocalPointGain(void** gain, float x, float y, float z, uint8_t duty);
EXPORT_AUTD void AUTDBesselBeamGain(void** gain, float x, float y, float z, float n_x, float n_y, float n_z, float theta_z, uint8_t duty);
EXPORT_AUTD void AUTDPlaneWaveGain(void** gain, float n_x, float n_y, float n_z, uint8_t duty);
EXPORT_AUTD void AUTDCustomGain(void** gain, uint16_t* data, int32_t data_length);
EXPORT_AUTD void AUTDTransducerTestGain(void** gain, int32_t idx, uint8_t duty, uint8_t phase);
#pragma endregion

#pragma region Modulation
EXPORT_AUTD void AUTDModulation(void** mod, uint8_t amp);
EXPORT_AUTD void AUTDDeleteModulation(void* mod);
EXPORT_AUTD void AUTDCustomModulation(void** mod, uint8_t* buf, uint32_t size);
EXPORT_AUTD void AUTDSawModulation(void** mod, int32_t freq);
EXPORT_AUTD void AUTDSineModulation(void** mod, int32_t freq, float amp, float offset);
EXPORT_AUTD void AUTDSquareModulation(void** mod, int32_t freq, uint8_t low, uint8_t high);
#pragma endregion

#pragma region Sequence
EXPORT_AUTD void AUTDSequence(void** out);
EXPORT_AUTD bool AUTDSequenceAddPoint(void* seq, float x, float y, float z);
EXPORT_AUTD bool AUTDSequenceAddPoints(void* seq, float* points, uint64_t size);
EXPORT_AUTD float AUTDSequenceSetFreq(void* seq, float freq);
EXPORT_AUTD float AUTDSequenceFreq(void* seq);
EXPORT_AUTD float AUTDSequenceSamplingFreq(void* seq);
EXPORT_AUTD uint16_t AUTDSequenceSamplingFreqDiv(void* seq);
EXPORT_AUTD void AUTDDeleteSequence(void* seq);
EXPORT_AUTD void AUTDCircumSequence(void** out, float x, float y, float z, float nx, float ny, float nz, float radius, uint64_t n);
#pragma endredion

#pragma region LowLevelInterface
EXPORT_AUTD bool AUTDAppendGain(void* handle, void* gain);
EXPORT_AUTD bool AUTDAppendGainSync(void* handle, void* gain, bool wait_for_send);
EXPORT_AUTD bool AUTDAppendModulation(void* handle, void* mod);
EXPORT_AUTD bool AUTDAppendModulationSync(void* handle, void* mod);
EXPORT_AUTD void AUTDAddSTMGain(void* handle, void* gain);
EXPORT_AUTD bool AUTDStartSTModulation(void* handle, float freq);
EXPORT_AUTD bool AUTDStopSTModulation(void* handle);
EXPORT_AUTD bool AUTDFinishSTModulation(void* handle);
EXPORT_AUTD bool AUTDAppendSequence(void* handle, void* seq);
EXPORT_AUTD void AUTDFlush(void* handle);
EXPORT_AUTD int32_t AUTDDeviceIdxForTransIdx(void* handle, int32_t global_trans_idx);
EXPORT_AUTD void AUTDTransPositionByGlobal(void* handle, int32_t global_trans_idx, float* x, float* y, float* z);
EXPORT_AUTD void AUTDTransPositionByLocal(void* handle, int32_t device_idx, int32_t local_trans_idx, float* x, float* y, float* z);
EXPORT_AUTD void AUTDDeviceDirection(void* handle, int32_t device_idx, float* x, float* y, float* z);
#pragma endregion
}
