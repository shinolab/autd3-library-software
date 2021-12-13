// File: autd3_c_api.h
// Project: include
// Created Date: 07/02/2018
// Author: Shun Suzuki
// -----
// Last Modified: 13/12/2021
// Modified By: Shun Suzuki (suzuki@hapis.k.u-tokyo.ac.jp)
// -----
// Copyright (c) 2018-2020 Hapis Lab. All rights reserved.
//

#pragma once

#include "./header.h"

#ifdef __cplusplus
extern "C" {
#endif
EXPORT_AUTD void AUTDCreateController(void** out);
EXPORT_AUTD bool_t AUTDOpenController(void* handle, void* p_link);
EXPORT_AUTD int32_t AUTDAddDevice(void* handle, double x, double y, double z, double rz1, double ry, double rz2);
EXPORT_AUTD int32_t AUTDAddDeviceQuaternion(void* handle, double x, double y, double z, double qw, double qx, double qy, double qz);
EXPORT_AUTD int32_t AUTDCloseController(void* handle);
EXPORT_AUTD int32_t AUTDClear(void* handle);
EXPORT_AUTD void AUTDFreeController(const void* handle);
EXPORT_AUTD bool_t AUTDIsOpen(const void* handle);
EXPORT_AUTD bool_t AUTDGetOutputEnable(const void* handle);
EXPORT_AUTD bool_t AUTDGetSilentMode(const void* handle);
EXPORT_AUTD bool_t AUTDGetForceFan(const void* handle);
EXPORT_AUTD bool_t AUTDGetReadsFPGAInfo(const void* handle);
EXPORT_AUTD bool_t AUTDGetOutputBalance(const void* handle);
EXPORT_AUTD bool_t AUTDGetCheckAck(const void* handle);
EXPORT_AUTD void AUTDSetOutputEnable(void* handle, bool_t enable);
EXPORT_AUTD void AUTDSetSilentMode(void* handle, bool_t mode);
EXPORT_AUTD void AUTDSetReadsFPGAInfo(void* handle, bool_t reads_fpga_info);
EXPORT_AUTD void AUTDSetOutputBalance(void* handle, bool_t output_balance);
EXPORT_AUTD void AUTDSetCheckAck(void* handle, bool_t check_ack);
EXPORT_AUTD void AUTDSetForceFan(void* handle, bool_t force);
EXPORT_AUTD double AUTDGetWavelength(const void* handle);
EXPORT_AUTD double AUTDGetAttenuation(const void* handle);
EXPORT_AUTD void AUTDSetWavelength(void* handle, double wavelength);
EXPORT_AUTD void AUTDSetAttenuation(void* handle, double attenuation);
EXPORT_AUTD bool_t AUTDGetFPGAInfo(void* handle, uint8_t* out);
EXPORT_AUTD int32_t AUTDUpdateCtrlFlags(void* handle);
EXPORT_AUTD int32_t AUTDSetDelayOffset(void* handle, const uint8_t* delay, const uint8_t* offset);
EXPORT_AUTD int32_t AUTDGetLastError(char* error);
EXPORT_AUTD int32_t AUTDNumDevices(const void* handle);
EXPORT_AUTD void AUTDTransPosition(const void* handle, int32_t device_idx, int32_t local_trans_idx, double* x, double* y, double* z);
EXPORT_AUTD void AUTDDeviceXDirection(const void* handle, int32_t device_idx, double* x, double* y, double* z);
EXPORT_AUTD void AUTDDeviceYDirection(const void* handle, int32_t device_idx, double* x, double* y, double* z);
EXPORT_AUTD void AUTDDeviceZDirection(const void* handle, int32_t device_idx, double* x, double* y, double* z);
EXPORT_AUTD int32_t AUTDGetFirmwareInfoListPointer(void* handle, void** out);
EXPORT_AUTD void AUTDGetFirmwareInfo(const void* p_firm_info_list, int32_t index, char* cpu_ver, char* fpga_ver);
EXPORT_AUTD void AUTDFreeFirmwareInfoListPointer(const void* p_firm_info_list);
EXPORT_AUTD void AUTDGainNull(void** gain);
EXPORT_AUTD void AUTDGainGrouped(void** gain, const void* handle);
EXPORT_AUTD void AUTDGainGroupedAdd(void* grouped_gain, int32_t device_id, void* gain);
EXPORT_AUTD void AUTDGainFocalPoint(void** gain, double x, double y, double z, uint8_t duty);
EXPORT_AUTD void AUTDGainBesselBeam(void** gain, double x, double y, double z, double n_x, double n_y, double n_z, double theta_z, uint8_t duty);
EXPORT_AUTD void AUTDGainPlaneWave(void** gain, double n_x, double n_y, double n_z, uint8_t duty);
EXPORT_AUTD void AUTDGainCustom(void** gain, const uint16_t* data, int32_t data_length);
EXPORT_AUTD void AUTDGainTransducerTest(void** gain, int32_t idx, uint8_t duty, uint8_t phase);
EXPORT_AUTD void AUTDDeleteGain(const void* gain);
EXPORT_AUTD void AUTDModulationStatic(void** mod, uint8_t duty);
EXPORT_AUTD void AUTDModulationCustom(void** mod, const uint8_t* buf, uint32_t size, uint32_t freq_div);
EXPORT_AUTD void AUTDModulationSine(void** mod, int32_t freq, double amp, double offset);
EXPORT_AUTD void AUTDModulationSineSquared(void** mod, int32_t freq, double amp, double offset);
EXPORT_AUTD void AUTDModulationSineLegacy(void** mod, double freq, double amp, double offset);
EXPORT_AUTD void AUTDModulationSquare(void** mod, int32_t freq, uint8_t low, uint8_t high, double duty);
EXPORT_AUTD uint32_t AUTDModulationSamplingFreqDiv(const void* mod);
EXPORT_AUTD void AUTDModulationSetSamplingFreqDiv(void* mod, uint32_t freq_div);
EXPORT_AUTD double AUTDModulationSamplingFreq(const void* mod);
EXPORT_AUTD void AUTDDeleteModulation(const void* mod);
EXPORT_AUTD void AUTDSequence(void** out);
EXPORT_AUTD void AUTDGainSequence(void** out, const void* handle, uint16_t gain_mode);
EXPORT_AUTD bool_t AUTDSequenceAddPoint(void* seq, double x, double y, double z, uint8_t duty);
EXPORT_AUTD bool_t AUTDSequenceAddPoints(void* seq, const double* points, uint64_t points_size, const uint8_t* duties, uint64_t duties_size);
EXPORT_AUTD bool_t AUTDSequenceAddGain(void* seq, void* gain);
EXPORT_AUTD double AUTDSequenceSetFreq(void* seq, double freq);
EXPORT_AUTD double AUTDSequenceFreq(const void* seq);
EXPORT_AUTD uint32_t AUTDSequencePeriod(const void* seq);
EXPORT_AUTD uint32_t AUTDSequenceSamplingPeriod(const void* seq);
EXPORT_AUTD double AUTDSequenceSamplingFreq(const void* seq);
EXPORT_AUTD uint32_t AUTDSequenceSamplingFreqDiv(const void* seq);
EXPORT_AUTD void AUTDSequenceSetSamplingFreqDiv(void* seq, uint32_t freq_div);
EXPORT_AUTD void AUTDCircumSequence(void** out, double x, double y, double z, double nx, double ny, double nz, double radius, uint64_t n);
EXPORT_AUTD void AUTDDeleteSequence(const void* seq);
EXPORT_AUTD int32_t AUTDStop(void* handle);
EXPORT_AUTD int32_t AUTDPause(void* handle);
EXPORT_AUTD int32_t AUTDResume(void* handle);
EXPORT_AUTD int32_t AUTDSendHeader(void* handle, void* header);
EXPORT_AUTD int32_t AUTDSendBody(void* handle, void* body);
EXPORT_AUTD int32_t AUTDSendHeaderBody(void* handle, void* header, void* body);
EXPORT_AUTD void AUTDSTMController(void** out, void* handle);
EXPORT_AUTD bool_t AUTDAddSTMGain(const void* handle, void* gain);
EXPORT_AUTD bool_t AUTDStartSTM(void* handle, double freq);
EXPORT_AUTD bool_t AUTDStopSTM(void* handle);
EXPORT_AUTD bool_t AUTDFinishSTM(void* handle);
#ifdef __cplusplus
}
#endif
