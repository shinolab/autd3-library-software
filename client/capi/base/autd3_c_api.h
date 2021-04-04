// File: autd3_c_api.h
// Project: include
// Created Date: 07/02/2018
// Author: Shun Suzuki
// -----
// Last Modified: 04/04/2021
// Modified By: Shun Suzuki (suzuki@hapis.k.u-tokyo.ac.jp)
// -----
// Copyright (c) 2018-2020 Hapis Lab. All rights reserved.
//

#pragma once

#include "autd_types.hpp"

#if WIN32
#define EXPORT_AUTD __declspec(dllexport)
#else
#define EXPORT_AUTD __attribute__((visibility("default")))
#endif

extern "C" {

#pragma region Controller
EXPORT_AUTD void AUTDCreateController(void** out);
EXPORT_AUTD bool AUTDOpenControllerWith(void* handle, void* p_link);
EXPORT_AUTD int32_t AUTDAddDevice(void* handle, autd::Float x, autd::Float y, autd::Float z, autd::Float rz1, autd::Float ry, autd::Float rz2,
                                  int32_t group_id);
EXPORT_AUTD int32_t AUTDAddDeviceQuaternion(void* handle, autd::Float x, autd::Float y, autd::Float z, autd::Float qua_w, autd::Float qua_x,
                                            autd::Float qua_y, autd::Float qua_z, int32_t group_id);
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
EXPORT_AUTD void AUTDGetLastError(char* error);
#pragma endregion

#pragma region Property
EXPORT_AUTD bool AUTDIsOpen(void* handle);
EXPORT_AUTD bool AUTDIsSilentMode(void* handle);
EXPORT_AUTD autd::Float AUTDWavelength(void* handle);
EXPORT_AUTD void AUTDSetWavelength(void* handle, autd::Float wavelength);
EXPORT_AUTD int32_t AUTDNumDevices(void* handle);
EXPORT_AUTD int32_t AUTDNumTransducers(void* handle);
EXPORT_AUTD uint64_t AUTDRemainingInBuffer(void* handle);
#pragma endregion

#pragma region Gain
EXPORT_AUTD void AUTDNullGain(void** gain);
EXPORT_AUTD void AUTDGroupedGain(void** gain, const int32_t* group_ids, void* const* in_gains, int32_t size);
EXPORT_AUTD void AUTDDeleteGain(void* gain);
EXPORT_AUTD void AUTDFocalPointGain(void** gain, float x, float y, float z, uint8_t duty);
EXPORT_AUTD void AUTDBesselBeamGain(void** gain, float x, float y, float z, float n_x, float n_y, float n_z, float theta_z, uint8_t duty);
EXPORT_AUTD void AUTDPlaneWaveGain(void** gain, float n_x, float n_y, float n_z, uint8_t duty);
EXPORT_AUTD void AUTDCustomGain(void** gain, const uint16_t* data, int32_t data_length);
EXPORT_AUTD void AUTDTransducerTestGain(void** gain, int32_t idx, uint8_t duty, uint8_t phase);
#pragma endregion

#pragma region Modulation
EXPORT_AUTD void AUTDModulation(void** mod, uint8_t amp);
EXPORT_AUTD void AUTDDeleteModulation(void* mod);
EXPORT_AUTD void AUTDCustomModulation(void** mod, const uint8_t* buf, uint32_t size);
EXPORT_AUTD void AUTDSawModulation(void** mod, int32_t freq);
EXPORT_AUTD void AUTDSineModulation(void** mod, int32_t freq, float amp, float offset);
EXPORT_AUTD void AUTDSquareModulation(void** mod, int32_t freq, uint8_t low, uint8_t high);
#pragma endregion

#pragma region Sequence
EXPORT_AUTD void AUTDSequence(void** out);
EXPORT_AUTD bool AUTDSequenceAppendPoint(void* seq, autd::Float x, autd::Float y, autd::Float z);
EXPORT_AUTD bool AUTDSequenceAppendPoints(void* seq, const autd::Float* points, uint64_t size);
EXPORT_AUTD autd::Float AUTDSequenceSetFreq(void* seq, autd::Float freq);
EXPORT_AUTD autd::Float AUTDSequenceFreq(void* seq);
EXPORT_AUTD autd::Float AUTDSequenceSamplingFreq(void* seq);
EXPORT_AUTD uint16_t AUTDSequenceSamplingFreqDiv(void* seq);
EXPORT_AUTD void AUTDDeleteSequence(void* seq);
EXPORT_AUTD void AUTDCircumSequence(void** out, autd::Float x, autd::Float y, autd::Float z, autd::Float nx, autd::Float ny, autd::Float nz,
                                    autd::Float radius, uint64_t n);
#pragma endredion

#pragma region LowLevelInterface
EXPORT_AUTD bool AUTDAppendGain(void* handle, void* gain);
EXPORT_AUTD bool AUTDAppendGainSync(void* handle, void* gain, bool wait_for_send);
EXPORT_AUTD bool AUTDAppendModulation(void* handle, void* mod);
EXPORT_AUTD bool AUTDAppendModulationSync(void* handle, void* mod);
EXPORT_AUTD void AUTDAppendSTMGain(void* handle, void* gain);
EXPORT_AUTD bool AUTDStartSTModulation(void* handle, autd::Float freq);
EXPORT_AUTD bool AUTDStopSTModulation(void* handle);
EXPORT_AUTD bool AUTDFinishSTModulation(void* handle);
EXPORT_AUTD bool AUTDAppendSequence(void* handle, void* seq);
EXPORT_AUTD void AUTDFlush(void* handle);
EXPORT_AUTD int32_t AUTDDeviceIdxForTransIdx(void* handle, int32_t global_trans_idx);
EXPORT_AUTD autd::Float* AUTDTransPositionByGlobal(void* handle, int32_t global_trans_idx);
EXPORT_AUTD autd::Float* AUTDTransPositionByLocal(void* handle, int32_t device_idx, int32_t local_trans_idx);
EXPORT_AUTD autd::Float* AUTDDeviceDirection(void* handle, int32_t device_idx);
#pragma endregion
}
