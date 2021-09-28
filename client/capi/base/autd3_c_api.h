// File: autd3_c_api.h
// Project: include
// Created Date: 07/02/2018
// Author: Shun Suzuki
// -----
// Last Modified: 28/09/2021
// Modified By: Shun Suzuki (suzuki@hapis.k.u-tokyo.ac.jp)
// -----
// Copyright (c) 2018-2020 Hapis Lab. All rights reserved.
//

#pragma once

#include "./header.h"

extern "C" {
EXPORT_AUTD void AUTDCreateController(void** out);
EXPORT_AUTD bool AUTDOpenController(const void* handle, void* p_link);
EXPORT_AUTD int32_t AUTDAddDevice(const void* handle, double x, double y, double z, double rz1, double ry, double rz2, int32_t gid);
EXPORT_AUTD int32_t AUTDAddDeviceQuaternion(const void* handle, double x, double y, double z, double qw, double qx, double qy, double qz,
                                            int32_t gid);
EXPORT_AUTD int32_t AUTDDeleteDevice(const void* handle, int32_t idx);
EXPORT_AUTD void AUTDClearDevices(const void* handle);
EXPORT_AUTD int32_t AUTDCloseController(const void* handle);
EXPORT_AUTD int32_t AUTDClear(const void* handle);
EXPORT_AUTD void AUTDFreeController(const void* handle);
EXPORT_AUTD bool AUTDIsOpen(const void* handle);
EXPORT_AUTD bool AUTDIsSilentMode(const void* handle);
EXPORT_AUTD bool AUTDIsForceFan(const void* handle);
EXPORT_AUTD bool AUTDIsReadsFPGAInfo(const void* handle);
EXPORT_AUTD bool AUTDIsOutputBalance(const void* handle);
EXPORT_AUTD bool AUTDIsCheckAck(const void* handle);
EXPORT_AUTD void AUTDSetSilentMode(const void* handle, bool mode);
EXPORT_AUTD void AUTDSetReadsFPGAInfo(const void* handle, bool reads_fpga_info);
EXPORT_AUTD void AUTDSetOutputBalance(const void* handle, bool output_balance);
EXPORT_AUTD void AUTDSetCheckAck(const void* handle, bool check_ack);
EXPORT_AUTD void AUTDSetForceFan(const void* handle, bool force);
EXPORT_AUTD double AUTDGetWavelength(const void* handle);
EXPORT_AUTD double AUTDGetAttenuation(const void* handle);
EXPORT_AUTD void AUTDSetWavelength(const void* handle, double wavelength);
EXPORT_AUTD void AUTDSetAttenuation(const void* handle, double attenuation);
EXPORT_AUTD bool AUTDGetFPGAInfo(const void* handle, uint8_t* out);
EXPORT_AUTD int32_t AUTDUpdateCtrlFlags(const void* handle);
EXPORT_AUTD int32_t AUTDSetOutputDelay(const void* handle, const uint8_t* delay);
EXPORT_AUTD int32_t AUTDSetDutyOffset(const void* handle, const uint8_t* offset);
EXPORT_AUTD int32_t AUTDSetDelayOffset(const void* handle, const uint8_t* delay, const uint8_t* offset);
EXPORT_AUTD int32_t AUTDGetLastError(char* error);
EXPORT_AUTD int32_t AUTDNumDevices(const void* handle);
EXPORT_AUTD int32_t AUTDNumTransducers(const void* handle);
EXPORT_AUTD int32_t AUTDDeviceIdxForTransIdx(const void* handle, int32_t global_trans_idx);
EXPORT_AUTD void AUTDTransPositionByGlobal(const void* handle, int32_t global_trans_idx, double* x, double* y, double* z);
EXPORT_AUTD void AUTDTransPositionByLocal(const void* handle, int32_t device_idx, int32_t local_trans_idx, double* x, double* y, double* z);
EXPORT_AUTD void AUTDDeviceXDirection(const void* handle, int32_t device_idx, double* x, double* y, double* z);
EXPORT_AUTD void AUTDDeviceYDirection(const void* handle, int32_t device_idx, double* x, double* y, double* z);
EXPORT_AUTD void AUTDDeviceZDirection(const void* handle, int32_t device_idx, double* x, double* y, double* z);
EXPORT_AUTD int32_t AUTDGetFirmwareInfoListPointer(const void* handle, void** out);
EXPORT_AUTD void AUTDGetFirmwareInfo(const void* p_firm_info_list, int32_t index, char* cpu_ver, char* fpga_ver);
EXPORT_AUTD void AUTDFreeFirmwareInfoListPointer(const void* p_firm_info_list);
EXPORT_AUTD void AUTDGainNull(void** gain);
EXPORT_AUTD void AUTDGainGrouped(void** gain);
EXPORT_AUTD void AUTDGainGroupedAdd(const void* grouped_gain, int32_t id, const void* gain);
EXPORT_AUTD void AUTDGainFocalPoint(void** gain, double x, double y, double z, uint8_t duty);
EXPORT_AUTD void AUTDGainBesselBeam(void** gain, double x, double y, double z, double n_x, double n_y, double n_z, double theta_z, uint8_t duty);
EXPORT_AUTD void AUTDGainPlaneWave(void** gain, double n_x, double n_y, double n_z, uint8_t duty);
EXPORT_AUTD void AUTDGainCustom(void** gain, const uint16_t* data, int32_t data_length);
EXPORT_AUTD void AUTDGainTransducerTest(void** gain, int32_t idx, uint8_t duty, uint8_t phase);
EXPORT_AUTD void AUTDDeleteGain(const void* gain);
EXPORT_AUTD void AUTDModulationStatic(void** mod, uint8_t duty);
EXPORT_AUTD void AUTDModulationCustom(void** mod, const uint8_t* buf, uint32_t size);
EXPORT_AUTD void AUTDModulationSine(void** mod, int32_t freq, double amp, double offset);
EXPORT_AUTD void AUTDModulationSinePressure(void** mod, int32_t freq, double amp, double offset);
EXPORT_AUTD void AUTDModulationSineLegacy(void** mod, double freq, double amp, double offset);
EXPORT_AUTD void AUTDModulationSquare(void** mod, int32_t freq, uint8_t low, uint8_t high, double duty);
EXPORT_AUTD void AUTDDeleteModulation(const void* mod);
EXPORT_AUTD void AUTDSequence(void** out);
EXPORT_AUTD void AUTDGainSequence(void** out, uint16_t gain_mode);
EXPORT_AUTD bool AUTDSequenceAddPoint(const void* seq, double x, double y, double z, uint8_t duty);
EXPORT_AUTD bool AUTDSequenceAddPoints(const void* seq, const double* points, uint64_t points_size, const uint8_t* duties, uint64_t duties_size);
EXPORT_AUTD bool AUTDSequenceAddGain(const void* seq, const void* gain);
EXPORT_AUTD double AUTDSequenceSetFreq(const void* seq, double freq);
EXPORT_AUTD double AUTDSequenceFreq(const void* seq);
EXPORT_AUTD uint32_t AUTDSequencePeriod(const void* seq);
EXPORT_AUTD uint32_t AUTDSequenceSamplingPeriod(const void* seq);
EXPORT_AUTD double AUTDSequenceSamplingFreq(const void* seq);
EXPORT_AUTD uint16_t AUTDSequenceSamplingFreqDiv(const void* seq);
EXPORT_AUTD void AUTDCircumSequence(void** out, double x, double y, double z, double nx, double ny, double nz, double radius, uint64_t n);
EXPORT_AUTD void AUTDDeleteSequence(const void* seq);
EXPORT_AUTD int32_t AUTDStop(const void* handle);
EXPORT_AUTD int32_t AUTDPause(const void* handle);
EXPORT_AUTD int32_t AUTDResume(const void* handle);
EXPORT_AUTD int32_t AUTDSendGain(const void* handle, const void* gain);
EXPORT_AUTD int32_t AUTDSendModulation(const void* handle, const void* mod);
EXPORT_AUTD int32_t AUTDSendGainModulation(const void* handle, const void* gain, const void* mod);
EXPORT_AUTD int32_t AUTDSendSequence(const void* handle, const void* seq);
EXPORT_AUTD int32_t AUTDSendGainSequence(const void* handle, const void* seq);
EXPORT_AUTD void AUTDSTMController(void** out, const void* handle);
EXPORT_AUTD bool AUTDAddSTMGain(const void* handle, const void* gain);
EXPORT_AUTD bool AUTDStartSTM(const void* handle, double freq);
EXPORT_AUTD bool AUTDStopSTM(const void* handle);
EXPORT_AUTD bool AUTDFinishSTM(const void* handle);
}
