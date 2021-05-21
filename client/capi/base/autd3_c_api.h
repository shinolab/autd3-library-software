// File: autd3_c_api.h
// Project: include
// Created Date: 07/02/2018
// Author: Shun Suzuki
// -----
// Last Modified: 21/05/2021
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
EXPORT_AUTD bool AUTDSynchronize(void* handle, uint16_t mod_smpl_freq_div, uint16_t mod_buf_size);
EXPORT_AUTD bool AUTDCloseController(void* handle);
EXPORT_AUTD bool AUTDClear(void* handle);
EXPORT_AUTD void AUTDFreeController(void* handle);
EXPORT_AUTD bool AUTDIsOpen(void* handle);
EXPORT_AUTD bool AUTDIsSilentMode(void* handle);
EXPORT_AUTD void AUTDSetSilentMode(void* handle, bool mode);
EXPORT_AUTD void AUTDSetReadFPGAInfo(void* handle, bool reads_fpga_info);
EXPORT_AUTD double AUTDWavelength(void* handle);
EXPORT_AUTD void AUTDSetWavelength(void* handle, double wavelength);
EXPORT_AUTD bool AUTDReadFPGAInfo(void* handle, uint8_t* out);
EXPORT_AUTD bool AUTDUpdateCtrlFlags(void* handle);
EXPORT_AUTD bool AUTDSetOutputDelay(void* handle, uint16_t* delay);

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

EXPORT_AUTD void AUTDNullGain(void** gain);
EXPORT_AUTD void AUTDGroupedGain(void** gain);
EXPORT_AUTD void AUTDGroupedGainAdd(void* grouped_gain, int32_t id, void* gain);
EXPORT_AUTD void AUTDFocalPointGain(void** gain, double x, double y, double z, uint8_t duty);
EXPORT_AUTD void AUTDBesselBeamGain(void** gain, double x, double y, double z, double n_x, double n_y, double n_z, double theta_z, uint8_t duty);
EXPORT_AUTD void AUTDPlaneWaveGain(void** gain, double n_x, double n_y, double n_z, uint8_t duty);
EXPORT_AUTD void AUTDCustomGain(void** gain, uint16_t* data, int32_t data_length);
EXPORT_AUTD void AUTDTransducerTestGain(void** gain, int32_t idx, uint8_t duty, uint8_t phase);
EXPORT_AUTD void AUTDDeleteGain(void* gain);

EXPORT_AUTD void AUTDStaticModulation(void** mod, uint8_t amp);
EXPORT_AUTD void AUTDCustomModulation(void** mod, uint8_t* buf, uint32_t size);
EXPORT_AUTD void AUTDSawModulation(void** mod, int32_t freq);
EXPORT_AUTD void AUTDSineModulation(void** mod, int32_t freq, double amp, double offset);
EXPORT_AUTD void AUTDSquareModulation(void** mod, int32_t freq, uint8_t low, uint8_t high);
EXPORT_AUTD void AUTDDeleteModulation(void* mod);

EXPORT_AUTD void AUTDSequence(void** out);
EXPORT_AUTD bool AUTDSequenceAddPoint(void* seq, double x, double y, double z);
EXPORT_AUTD bool AUTDSequenceAddPoints(void* seq, double* points, uint64_t size);
EXPORT_AUTD double AUTDSequenceSetFreq(void* seq, double freq);
EXPORT_AUTD double AUTDSequenceFreq(void* seq);
EXPORT_AUTD uint32_t AUTDSequencePeriod(void* seq);
EXPORT_AUTD uint32_t AUTDSequenceSamplingPeriod(void* seq);
EXPORT_AUTD double AUTDSequenceSamplingFreq(void* seq);
EXPORT_AUTD uint16_t AUTDSequenceSamplingFreqDiv(void* seq);
EXPORT_AUTD void AUTDDeleteSequence(void* seq);
EXPORT_AUTD void AUTDCircumSequence(void** out, double x, double y, double z, double nx, double ny, double nz, double radius, uint64_t n);

EXPORT_AUTD bool AUTDStop(void* handle);
EXPORT_AUTD bool AUTDSendGain(void* handle, void* gain);
EXPORT_AUTD bool AUTDSendModulation(void* handle, void* mod);
EXPORT_AUTD bool AUTDSendGainModulation(void* handle, void* gain, void* mod);
EXPORT_AUTD void AUTDAddSTMGain(void* handle, void* gain);
EXPORT_AUTD bool AUTDStartSTM(void* handle, double freq);
EXPORT_AUTD bool AUTDStopSTM(void* handle);
EXPORT_AUTD bool AUTDFinishSTM(void* handle);
}
