// File: autd3_c_api.h
// Project: include
// Created Date: 07/02/2018
// Author: Shun Suzuki
// -----
// Last Modified: 20/02/2021
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

#define VOID_PTR void*

extern "C" {

#pragma region Controller
EXPORT_AUTD void AUTDCreateController(VOID_PTR* out);
EXPORT_AUTD int32_t AUTDOpenControllerWith(VOID_PTR handle, VOID_PTR p_link);
EXPORT_AUTD int32_t AUTDAddDevice(VOID_PTR handle, float x, float y, float z, float rz1, float ry, float rz2, int32_t group_id);
EXPORT_AUTD int32_t AUTDAddDeviceQuaternion(VOID_PTR handle, float x, float y, float z, float qua_w, float qua_x, float qua_y, float qua_z,
                                            int32_t group_id);
EXPORT_AUTD bool AUTDCalibrate(VOID_PTR handle, int32_t smpl_freq, int32_t buf_size);
EXPORT_AUTD void AUTDCloseController(VOID_PTR handle);
EXPORT_AUTD void AUTDClear(VOID_PTR handle);
EXPORT_AUTD void AUTDFreeController(VOID_PTR handle);
EXPORT_AUTD void AUTDSetSilentMode(VOID_PTR handle, bool mode);
EXPORT_AUTD void AUTDStop(VOID_PTR handle);
EXPORT_AUTD int32_t AUTDGetFirmwareInfoListPointer(VOID_PTR handle, VOID_PTR* out);
EXPORT_AUTD void AUTDGetFirmwareInfo(VOID_PTR p_firm_info_list, int32_t index, char* cpu_ver, char* fpga_ver);
EXPORT_AUTD void AUTDFreeFirmwareInfoListPointer(VOID_PTR p_firm_info_list);
#pragma endregion

#pragma region Property
EXPORT_AUTD bool AUTDIsOpen(VOID_PTR handle);
EXPORT_AUTD bool AUTDIsSilentMode(VOID_PTR handle);
EXPORT_AUTD float AUTDWavelength(VOID_PTR handle);
EXPORT_AUTD void AUTDSetWavelength(VOID_PTR handle, float wavelength);
EXPORT_AUTD void AUTDSetDelay(VOID_PTR handle, const uint16_t* delay, int32_t data_length);
EXPORT_AUTD int32_t AUTDNumDevices(VOID_PTR handle);
EXPORT_AUTD int32_t AUTDNumTransducers(VOID_PTR handle);
EXPORT_AUTD uint64_t AUTDRemainingInBuffer(VOID_PTR handle);
#pragma endregion

#pragma region Gain
EXPORT_AUTD void AUTDNullGain(VOID_PTR* gain);
EXPORT_AUTD void AUTDGroupedGain(VOID_PTR* gain, const int32_t* group_ids, VOID_PTR const* in_gains, int32_t size);
EXPORT_AUTD void AUTDDeleteGain(VOID_PTR gain);
#pragma endregion

#pragma region Modulation
EXPORT_AUTD void AUTDModulation(VOID_PTR* mod, uint8_t amp);
EXPORT_AUTD void AUTDDeleteModulation(VOID_PTR mod);
#pragma endregion

#pragma region Sequence
EXPORT_AUTD void AUTDSequence(VOID_PTR* out);
EXPORT_AUTD void AUTDSequenceAppendPoint(VOID_PTR seq, float x, float y, float z);
EXPORT_AUTD void AUTDSequenceAppendPoints(VOID_PTR seq, const float* points, uint64_t size);
EXPORT_AUTD float AUTDSequenceSetFreq(VOID_PTR seq, float freq);
EXPORT_AUTD float AUTDSequenceFreq(VOID_PTR seq);
EXPORT_AUTD float AUTDSequenceSamplingFreq(VOID_PTR seq);
EXPORT_AUTD uint16_t AUTDSequenceSamplingFreqDiv(VOID_PTR seq);
EXPORT_AUTD void AUTDCircumSequence(VOID_PTR* out, float x, float y, float z, float nx, float ny, float nz, float radius, uint64_t n);
EXPORT_AUTD void AUTDDeleteSequence(VOID_PTR seq);
#pragma endredion

#pragma region LowLevelInterface
EXPORT_AUTD void AUTDAppendGain(VOID_PTR handle, VOID_PTR gain);
EXPORT_AUTD void AUTDAppendGainSync(VOID_PTR handle, VOID_PTR gain, bool wait_for_send);
EXPORT_AUTD void AUTDAppendModulation(VOID_PTR handle, VOID_PTR mod);
EXPORT_AUTD void AUTDAppendModulationSync(VOID_PTR handle, VOID_PTR mod);
EXPORT_AUTD void AUTDAppendSTMGain(VOID_PTR handle, VOID_PTR gain);
EXPORT_AUTD void AUTDStartSTModulation(VOID_PTR handle, float freq);
EXPORT_AUTD void AUTDStopSTModulation(VOID_PTR handle);
EXPORT_AUTD void AUTDFinishSTModulation(VOID_PTR handle);
EXPORT_AUTD void AUTDAppendSequence(VOID_PTR handle, VOID_PTR seq);
EXPORT_AUTD void AUTDFlush(VOID_PTR handle);
EXPORT_AUTD int32_t AUTDDeviceIdxForTransIdx(VOID_PTR handle, int32_t global_trans_idx);
EXPORT_AUTD float* AUTDTransPositionByGlobal(VOID_PTR handle, int32_t global_trans_idx);
EXPORT_AUTD float* AUTDTransPositionByLocal(VOID_PTR handle, int32_t device_idx, int32_t local_trans_idx);
EXPORT_AUTD float* AUTDDeviceDirection(VOID_PTR handle, int32_t device_idx);
#pragma endregion
}
