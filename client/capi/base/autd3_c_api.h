// File: autd3_c_api.h
// Project: include
// Created Date: 07/02/2018
// Author: Shun Suzuki
// -----
// Last Modified: 07/09/2021
// Modified By: Shun Suzuki (suzuki@hapis.k.u-tokyo.ac.jp)
// -----
// Copyright (c) 2018-2020 Hapis Lab. All rights reserved.
//

#pragma once

#include "./header.h"

extern "C" {

EXPORT_AUTD void AUTDCreateController(void** out);
EXPORT_AUTD bool AUTDOpenController(void* handle, void* p_link);
EXPORT_AUTD int32_t AUTDAddDevice(void* handle, double x, double y, double z, double rz1, double ry, double rz2, int32_t gid);
EXPORT_AUTD int32_t AUTDAddDeviceQuaternion(void* handle, double x, double y, double z, double qw, double qx, double qy, double qz, int32_t gid);
EXPORT_AUTD int32_t AUTDDeleteDevice(void* handle, int32_t idx);
EXPORT_AUTD void AUTDClearDevices(void* handle);
EXPORT_AUTD bool AUTDCloseController(void* handle);
EXPORT_AUTD bool AUTDClear(void* handle);
EXPORT_AUTD void AUTDFreeController(void* handle);
EXPORT_AUTD bool AUTDIsOpen(void* handle);
EXPORT_AUTD bool AUTDIsSilentMode(void* handle);
EXPORT_AUTD bool AUTDIsForceFan(void* handle);
EXPORT_AUTD bool AUTDIsReadsFPGAInfo(void* handle);
EXPORT_AUTD void AUTDSetSilentMode(void* handle, bool mode);
EXPORT_AUTD void AUTDSetReadsFPGAInfo(void* handle, bool reads_fpga_info);
EXPORT_AUTD void AUTDSetForceFan(void* handle, bool force);
EXPORT_AUTD double AUTDGetWavelength(void* handle);
EXPORT_AUTD double AUTDGetAttenuation(void* handle);
EXPORT_AUTD void AUTDSetWavelength(void* handle, double wavelength);
EXPORT_AUTD void AUTDSetAttenuation(void* handle, double attenuation);
EXPORT_AUTD bool AUTDGetFPGAInfo(void* handle, uint8_t* out);
EXPORT_AUTD bool AUTDUpdateCtrlFlags(void* handle);
EXPORT_AUTD bool AUTDSetOutputDelay(void* handle, const uint8_t* delay);
EXPORT_AUTD bool AUTDSetDutyOffset(void* handle, const uint8_t* offset);
EXPORT_AUTD bool AUTDSetDelayOffset(void* handle, const uint8_t* delay, const uint8_t* offset);

EXPORT_AUTD int32_t AUTDGetLastError(char* error);

EXPORT_AUTD int32_t AUTDNumDevices(void* handle);
EXPORT_AUTD int32_t AUTDNumTransducers(void* handle);
EXPORT_AUTD int32_t AUTDDeviceIdxForTransIdx(void* handle, int32_t global_trans_idx);
EXPORT_AUTD void AUTDTransPositionByGlobal(void* handle, int32_t global_trans_idx, double* x, double* y, double* z);
EXPORT_AUTD void AUTDTransPositionByLocal(void* handle, int32_t device_idx, int32_t local_trans_idx, double* x, double* y, double* z);
EXPORT_AUTD void AUTDDeviceXDirection(void* handle, int32_t device_idx, double* x, double* y, double* z);
EXPORT_AUTD void AUTDDeviceYDirection(void* handle, int32_t device_idx, double* x, double* y, double* z);
EXPORT_AUTD void AUTDDeviceZDirection(void* handle, int32_t device_idx, double* x, double* y, double* z);

EXPORT_AUTD int32_t AUTDGetFirmwareInfoListPointer(void* handle, void** out);
EXPORT_AUTD void AUTDGetFirmwareInfo(void* p_firm_info_list, int32_t index, char* cpu_ver, char* fpga_ver);
EXPORT_AUTD void AUTDFreeFirmwareInfoListPointer(void* p_firm_info_list);

EXPORT_AUTD void AUTDGainNull(void** gain);
EXPORT_AUTD void AUTDGainGrouped(void** gain);
EXPORT_AUTD void AUTDGainGroupedAdd(void* grouped_gain, int32_t id, void* gain);
EXPORT_AUTD void AUTDGainFocalPoint(void** gain, double x, double y, double z, uint8_t duty);
EXPORT_AUTD void AUTDGainBesselBeam(void** gain, double x, double y, double z, double n_x, double n_y, double n_z, double theta_z, uint8_t duty);
EXPORT_AUTD void AUTDGainPlaneWave(void** gain, double n_x, double n_y, double n_z, uint8_t duty);
EXPORT_AUTD void AUTDGainCustom(void** gain, const uint16_t* data, int32_t data_length);
EXPORT_AUTD void AUTDGainTransducerTest(void** gain, int32_t idx, uint8_t duty, uint8_t phase);
EXPORT_AUTD void AUTDDeleteGain(void* gain);

EXPORT_AUTD void AUTDModulationStatic(void** mod, uint8_t amp);
EXPORT_AUTD void AUTDModulationCustom(void** mod, uint8_t* buf, uint32_t size);
EXPORT_AUTD void AUTDModulationSine(void** mod, int32_t freq, double amp, double offset);
EXPORT_AUTD void AUTDModulationSinePressure(void** mod, int32_t freq, double amp, double offset);
EXPORT_AUTD void AUTDModulationSineLegacy(void** mod, double freq, double amp, double offset);
EXPORT_AUTD void AUTDModulationSquare(void** mod, int32_t freq, uint8_t low, uint8_t high);
EXPORT_AUTD void AUTDDeleteModulation(void* mod);

EXPORT_AUTD void AUTDSequence(void** out);
EXPORT_AUTD void AUTDGainSequence(void** out, uint16_t gain_mode);
EXPORT_AUTD bool AUTDSequenceAddPoint(void* seq, double x, double y, double z, uint8_t duty);
EXPORT_AUTD bool AUTDSequenceAddPoints(void* seq, const double* points, uint64_t points_size, uint8_t* duties, uint64_t duties_size);
EXPORT_AUTD bool AUTDSequenceAddGain(void* seq, void* gain);
EXPORT_AUTD double AUTDSequenceSetFreq(void* seq, double freq);
EXPORT_AUTD double AUTDSequenceFreq(void* seq);
EXPORT_AUTD uint32_t AUTDSequencePeriod(void* seq);
EXPORT_AUTD uint32_t AUTDSequenceSamplingPeriod(void* seq);
EXPORT_AUTD double AUTDSequenceSamplingFreq(void* seq);
EXPORT_AUTD uint16_t AUTDSequenceSamplingFreqDiv(void* seq);
EXPORT_AUTD void AUTDDeleteSequence(void* seq);
EXPORT_AUTD void AUTDCircumSequence(void** out, double x, double y, double z, double nx, double ny, double nz, double radius, uint64_t n);

EXPORT_AUTD bool AUTDStop(void* handle);
EXPORT_AUTD bool AUTDPause(void* handle);
EXPORT_AUTD bool AUTDResume(void* handle);
EXPORT_AUTD bool AUTDSendGain(void* handle, void* gain, bool wait_for_msg_processed);
EXPORT_AUTD bool AUTDSendModulation(void* handle, void* mod);
EXPORT_AUTD bool AUTDSendGainModulation(void* handle, void* gain, void* mod, bool wait_for_msg_processed);
EXPORT_AUTD bool AUTDSendSequence(void* handle, void* seq);
EXPORT_AUTD bool AUTDSendGainSequence(void* handle, void* seq);

EXPORT_AUTD void AUTDSTMController(void** out, void* handle);
EXPORT_AUTD bool AUTDAddSTMGain(void* handle, void* gain);
EXPORT_AUTD bool AUTDStartSTM(void* handle, double freq);
EXPORT_AUTD bool AUTDStopSTM(void* handle);
EXPORT_AUTD bool AUTDFinishSTM(void* handle);
}
