// File: c_api.cpp
// Project: capi
// Created Date: 02/07/2018
// Author: Shun Suzuki
// -----
// Last Modified: 30/04/2020
// Modified By: Shun Suzuki (suzuki@hapis.k.u-tokyo.ac.jp)
// -----
// Copyright (c) 2018-2020 Hapis Lab. All rights reserved.
//

#include <errno.h>

#if WIN32
#include <codeanalysis\warnings.h>
#pragma warning(push)
#pragma warning(disable : ALL_CODE_ANALYSIS_WARNINGS)
#endif
#include <Eigen/Geometry>
#if WIN32
#pragma warning(pop)
#endif

#include "./autd3_c_api.h"
#include "autd3.hpp"
#include "quaternion.hpp"
#include "vector3.hpp"

#pragma region Controller
void AUTDCreateController(AUTDControllerHandle *out) {
  auto *cnt = autd::Controller::Create();
  *out = cnt;
}
int AUTDOpenController(AUTDControllerHandle handle, int linkType, const char *location) {
  auto *cnt = static_cast<autd::Controller *>(handle);
  cnt->Open(static_cast<autd::LinkType>(linkType), std::string(location));
  if (!cnt->is_open()) return ENXIO;
  return 0;
}
int AUTDAddDevice(AUTDControllerHandle handle, double x, double y, double z, double rz1, double ry, double rz2, int group_id) {
  auto *cnt = static_cast<autd::Controller *>(handle);
  return cnt->geometry()->AddDevice(autd::Vector3(x, y, z), autd::Vector3(rz1, ry, rz2), group_id);
}
int AUTDAddDeviceQuaternion(AUTDControllerHandle handle, double x, double y, double z, double qua_w, double qua_x, double qua_y, double qua_z,
                            int group_id) {
  auto *cnt = static_cast<autd::Controller *>(handle);
  return cnt->geometry()->AddDeviceQuaternion(autd::Vector3(x, y, z), autd::Quaternion(qua_w, qua_x, qua_y, qua_z), group_id);
}
void AUTDDelDevice(AUTDControllerHandle handle, int devId) {
  auto *cnt = static_cast<autd::Controller *>(handle);
  cnt->geometry()->DelDevice(devId);
}
void AUTDCloseController(AUTDControllerHandle handle) {
  auto *cnt = static_cast<autd::Controller *>(handle);
  cnt->Close();
}
void AUTDFreeController(AUTDControllerHandle handle) {
  auto *cnt = static_cast<autd::Controller *>(handle);
  delete cnt;
}
void AUTDSetSilentMode(AUTDControllerHandle handle, bool mode) {
  auto *cnt = static_cast<autd::Controller *>(handle);
  cnt->SetSilentMode(mode);
}
bool AUTDCalibrateModulation(AUTDControllerHandle handle) {
  auto *cnt = static_cast<autd::Controller *>(handle);
  return cnt->CalibrateModulation();
}
void AUTDStop(AUTDControllerHandle handle) {
  auto *cnt = static_cast<autd::Controller *>(handle);
  cnt->Stop();
}
int AUTDGetAdapterPointer(void **out) {
  int size;
  auto adapters = autd::Controller::EnumerateAdapters(&size);
  *out = adapters;
  return size;
}
void AUTDGetAdapter(void *p_adapter, int index, char *desc, char *name) {
  auto *adapters = static_cast<std::pair<std::string, std::string> *>(p_adapter);
  auto desc_ = adapters[index].first;
  auto name_ = adapters[index].second;
  std::char_traits<char>::copy(desc, desc_.c_str(), desc_.size() + 1);
  std::char_traits<char>::copy(name, name_.c_str(), name_.size() + 1);
}
void AUTDFreeAdapterPointer(void *p_adapter) {
  auto *adapters = static_cast<std::pair<std::string, std::string> *>(p_adapter);
  delete[] adapters;
}
int AUTDGetFirmwareInfoListPointer(AUTDControllerHandle handle, void **out) {
  auto *cnt = static_cast<autd::Controller *>(handle);
  int size = cnt->geometry()->numDevices();
  auto list = cnt->firmware_info_list();
  *out = list;
  return size;
}
void AUTDGetFirmwareInfo(void *pfirminfolist, int index, char *cpu_ver, char *fpga_ver) {
  auto *list = static_cast<autd::FirmwareInfoList>(pfirminfolist);
  auto cpu_ver_ = list[index].cpu_version();
  auto fpga_ver_ = list[index].fpga_version();
  std::char_traits<char>::copy(cpu_ver, cpu_ver_.c_str(), cpu_ver_.size() + 1);
  std::char_traits<char>::copy(fpga_ver, fpga_ver_.c_str(), fpga_ver_.size() + 1);
}
void AUTDFreeFirmwareInfoListPointer(void *pfirminfolist) {
  auto *list = static_cast<autd::FirmwareInfoList>(pfirminfolist);
  delete[] list;
}
#pragma endregion

#pragma region Property
bool AUTDIsOpen(AUTDControllerHandle handle) {
  auto *cnt = static_cast<autd::Controller *>(handle);
  return cnt->is_open();
}
bool AUTDIsSilentMode(AUTDControllerHandle handle) {
  auto *cnt = static_cast<autd::Controller *>(handle);
  return cnt->silent_mode();
}
int AUTDNumDevices(AUTDControllerHandle handle) {
  auto *cnt = static_cast<autd::Controller *>(handle);
  return cnt->geometry()->numDevices();
}
int AUTDNumTransducers(AUTDControllerHandle handle) {
  auto *cnt = static_cast<autd::Controller *>(handle);
  return cnt->geometry()->numTransducers();
}
size_t AUTDRemainingInBuffer(AUTDControllerHandle handle) {
  auto *cnt = static_cast<autd::Controller *>(handle);
  return cnt->remainingInBuffer();
}
#pragma endregion

#pragma region Gain
void AUTDFocalPointGain(AUTDGainPtr *gain, double x, double y, double z, uint8_t amp) {
  auto *g = autd::FocalPointGain::Create(autd::Vector3(x, y, z), amp);
  *gain = g;
}
void AUTDGroupedGain(AUTDGainPtr *gain, int *group_ids, AUTDGainPtr *gains, int size) {
  std::map<int, autd::GainPtr> gainmap;

  for (int i = 0; i < size; i++) {
    auto id = group_ids[i];
    auto gain_id = gains[i];
    auto *g = static_cast<autd::GainPtr>(gain_id);
    gainmap[id] = g;
  }

  auto *ggain = autd::GroupedGain::Create(gainmap);

  *gain = ggain;
}
void AUTDBesselBeamGain(AUTDGainPtr *gain, double x, double y, double z, double n_x, double n_y, double n_z, double theta_z) {
  auto *g = autd::BesselBeamGain::Create(autd::Vector3(x, y, z), autd::Vector3(n_x, n_y, n_z), theta_z);
  *gain = g;
}
void AUTDPlaneWaveGain(AUTDGainPtr *gain, double n_x, double n_y, double n_z) {
  auto *g = autd::PlaneWaveGain::Create(autd::Vector3(n_x, n_y, n_z));
  *gain = g;
}
void AUTDCustomGain(AUTDGainPtr *gain, uint16_t *data, int data_length) {
  auto *g = autd::CustomGain::Create(data, data_length);
  *gain = g;
}
void AUTDHoloGain(AUTDGainPtr *gain, double *points, double *amps, int size) {
  std::vector<autd::Vector3> holo;
  std::vector<double> amps_;
  for (int i = 0; i < size; i++) {
    autd::Vector3 v(points[3 * i], points[3 * i + 1], points[3 * i + 2]);
    holo.push_back(v);
    amps_.push_back(amps[i]);
  }

  auto *g = autd::HoloGainSdp::Create(holo, amps_);
  *gain = g;
}
void AUTDTransducerTestGain(AUTDGainPtr *gain, int idx, int amp, int phase) {
  auto *g = autd::TransducerTestGain::Create(idx, amp, phase);
  *gain = g;
}
void AUTDNullGain(AUTDGainPtr *gain) {
  auto *g = autd::NullGain::Create();
  *gain = g;
}
void AUTDDeleteGain(AUTDGainPtr gain) {
  auto *g = static_cast<autd::GainPtr>(gain);
  delete g;
}
#pragma endregion

#pragma region Modulation
void AUTDModulation(AUTDModulationPtr *mod, uint8_t amp) {
  auto *m = autd::Modulation::Create(amp);
  *mod = m;
}
void AUTDRawPCMModulation(AUTDModulationPtr *mod, const char *filename, double samp_freq) {
  auto *m = autd::RawPCMModulation::Create(std::string(filename), samp_freq);
  *mod = m;
}
void AUTDSawModulation(AUTDModulationPtr *mod, int freq) {
  auto *m = autd::SawModulation::Create(freq);
  *mod = m;
}
void AUTDSineModulation(AUTDModulationPtr *mod, int freq, double amp, double offset) {
  auto *m = autd::SineModulation::Create(freq, amp, offset);
  *mod = m;
}
void AUTDWavModulation(AUTDModulationPtr *mod, const char *filename) {
  auto *m = autd::WavModulation::Create(std::string(filename));
  *mod = m;
}
void AUTDDeleteModulation(AUTDModulationPtr mod) {
  auto *m = static_cast<autd::Modulation *>(mod);
  delete m;
}
#pragma endregion

#pragma region LowLevelInterface
void AUTDAppendGain(AUTDControllerHandle handle, AUTDGainPtr gain) {
  auto *cnt = static_cast<autd::Controller *>(handle);
  auto *g = static_cast<autd::GainPtr>(gain);
  cnt->AppendGain(g);
}

void AUTDAppendGainSync(AUTDControllerHandle handle, AUTDGainPtr gain, bool wait_for_send) {
  auto *cnt = static_cast<autd::Controller *>(handle);
  auto *g = static_cast<autd::GainPtr>(gain);
  cnt->AppendGainSync(g, wait_for_send);
}
void AUTDAppendModulation(AUTDControllerHandle handle, AUTDModulationPtr mod) {
  auto *cnt = static_cast<autd::Controller *>(handle);
  auto *m = static_cast<autd::Modulation *>(mod);
  cnt->AppendModulation(m);
}
void AUTDAppendModulationSync(AUTDControllerHandle handle, AUTDModulationPtr mod) {
  auto *cnt = static_cast<autd::Controller *>(handle);
  auto *m = static_cast<autd::Modulation *>(mod);
  cnt->AppendModulationSync(m);
}
void AUTDAppendSTMGain(AUTDControllerHandle handle, AUTDGainPtr gain) {
  auto *cnt = static_cast<autd::Controller *>(handle);
  auto *g = static_cast<autd::GainPtr>(gain);
  cnt->AppendSTMGain(g);
}
void AUTDStartSTModulation(AUTDControllerHandle handle, double freq) {
  auto *cnt = static_cast<autd::Controller *>(handle);
  cnt->StartSTModulation(freq);
}
void AUTDStopSTModulation(AUTDControllerHandle handle) {
  auto *cnt = static_cast<autd::Controller *>(handle);
  cnt->StopSTModulation();
}
void AUTDFinishSTModulation(AUTDControllerHandle handle) {
  auto *cnt = static_cast<autd::Controller *>(handle);
  cnt->FinishSTModulation();
}
void AUTDFlush(AUTDControllerHandle handle) {
  auto *cnt = static_cast<autd::Controller *>(handle);
  cnt->Flush();
}
int AUTDDevIdForDeviceIdx(AUTDControllerHandle handle, int dev_idx) {
  auto *cnt = static_cast<autd::Controller *>(handle);
  return cnt->geometry()->deviceIdForDeviceIdx(dev_idx);
}
int AUTDDevIdForTransIdx(AUTDControllerHandle handle, int trans_idx) {
  auto *cnt = static_cast<autd::Controller *>(handle);
  return cnt->geometry()->deviceIdForTransIdx(trans_idx);
}
double *AUTDTransPosition(AUTDControllerHandle handle, int trans_idx) {
  auto *cnt = static_cast<autd::Controller *>(handle);
  auto pos = cnt->geometry()->position(trans_idx);
  auto *array = new double[3];
  array[0] = pos.x();
  array[1] = pos.y();
  array[2] = pos.z();
  return array;
}
double *AUTDTransDirection(AUTDControllerHandle handle, int trans_idx) {
  auto *cnt = static_cast<autd::Controller *>(handle);
  auto dir = cnt->geometry()->direction(trans_idx);
  auto *array = new double[3];
  array[0] = dir.x();
  array[1] = dir.y();
  array[2] = dir.z();
  return array;
}
#pragma endregion

#pragma region Debug
#ifdef UNITY_DEBUG
DebugLogFunc _debugLogFunc = nullptr;

void DebugLog(const char *msg) {
  if (_debugLogFunc != nullptr) _debugLogFunc(msg);
}

void SetDebugLog(DebugLogFunc func) { _debugLogFunc = func; }
#endif

#pragma endregion
