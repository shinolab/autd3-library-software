// File: c_api.cpp
// Project: capi
// Created Date: 02/07/2018
// Author: Shun Suzuki
// -----
// Last Modified: 19/05/2020
// Modified By: Shun Suzuki (suzuki@hapis.k.u-tokyo.ac.jp)
// -----
// Copyright (c) 2018-2020 Hapis Lab. All rights reserved.
//

#include <errno.h>

#include <cstdint>

#include "./autd3_c_api.h"
#include "autd3.hpp"
#include "emulator_link.hpp"
#include "ethercat_link.hpp"
#include "soem_link.hpp"

#pragma region Controller
void AUTDCreateController(AUTDControllerHandle *out) {
  auto *cnt = autd::Controller::Create();
  *out = cnt;
}
int32_t AUTDOpenController(AUTDControllerHandle handle, int32_t linkType, const char *location) {
  auto *cnt = static_cast<autd::ControllerPtr>(handle);
  cnt->Open(static_cast<autd::LinkType>(linkType), std::string(location));
  if (!cnt->is_open()) return ENXIO;
  return 0;
}
int32_t AUTDOpenControllerWith(AUTDControllerHandle handle, AUTDLinkPtr plink) {
  auto *cnt = static_cast<autd::ControllerPtr>(handle);
  auto *link = static_cast<autd::LinkPtr>(plink);
  cnt->OpenWith(link);
  if (!cnt->is_open()) return ENXIO;
  return 0;
}
int32_t AUTDAddDevice(AUTDControllerHandle handle, double x, double y, double z, double rz1, double ry, double rz2, int32_t group_id) {
  auto *cnt = static_cast<autd::ControllerPtr>(handle);
  return cnt->geometry()->AddDevice(autd::Vector3(x, y, z), autd::Vector3(rz1, ry, rz2), group_id);
}
int32_t AUTDAddDeviceQuaternion(AUTDControllerHandle handle, double x, double y, double z, double qua_w, double qua_x, double qua_y, double qua_z,
                                int32_t group_id) {
  auto *cnt = static_cast<autd::Controller *>(handle);
  return cnt->geometry()->AddDeviceQuaternion(autd::Vector3(x, y, z), autd::Quaternion(qua_w, qua_x, qua_y, qua_z), group_id);
}
void AUTDDelDevice(AUTDControllerHandle handle, int32_t devId) {
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
int32_t AUTDGetAdapterPointer(void **out) {
  int32_t size;
  auto adapters = autd::Controller::EnumerateAdapters(&size);
  *out = adapters;
  return size;
}
void AUTDGetAdapter(void *p_adapter, int32_t index, char *desc, char *name) {
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
int32_t AUTDGetFirmwareInfoListPointer(AUTDControllerHandle handle, void **out) {
  auto *cnt = static_cast<autd::Controller *>(handle);
  int32_t size = cnt->geometry()->numDevices();
  auto list = cnt->firmware_info_list();
  *out = list;
  return size;
}
void AUTDGetFirmwareInfo(void *pfirminfolist, int32_t index, char *cpu_ver, char *fpga_ver) {
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
int32_t AUTDNumDevices(AUTDControllerHandle handle) {
  auto *cnt = static_cast<autd::Controller *>(handle);
  return cnt->geometry()->numDevices();
}
int32_t AUTDNumTransducers(AUTDControllerHandle handle) {
  auto *cnt = static_cast<autd::Controller *>(handle);
  return cnt->geometry()->numTransducers();
}
uint64_t AUTDRemainingInBuffer(AUTDControllerHandle handle) {
  auto *cnt = static_cast<autd::Controller *>(handle);
  return static_cast<uint64_t>(cnt->remainingInBuffer());
}
#pragma endregion

#pragma region Gain
void AUTDFocalPointGain(AUTDGainPtr *gain, double x, double y, double z, uint8_t amp) {
  auto *g = autd::FocalPointGain::Create(autd::Vector3(x, y, z), amp);
  *gain = g;
}
void AUTDGroupedGain(AUTDGainPtr *gain, int32_t *group_ids, AUTDGainPtr *gains, int32_t size) {
  std::map<int, autd::GainPtr> gainmap;

  for (int32_t i = 0; i < size; i++) {
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
void AUTDCustomGain(AUTDGainPtr *gain, uint16_t *data, int32_t data_length) {
  auto *g = autd::CustomGain::Create(data, data_length);
  *gain = g;
}
void AUTDHoloGain(AUTDGainPtr *gain, double *points, double *amps, int32_t size) {
  std::vector<autd::Vector3> holo;
  std::vector<double> amps_;
  for (int32_t i = 0; i < size; i++) {
    autd::Vector3 v(points[3 * i], points[3 * i + 1], points[3 * i + 2]);
    holo.push_back(v);
    amps_.push_back(amps[i]);
  }

  auto *g = autd::HoloGainSdp::Create(holo, amps_);
  *gain = g;
}
void AUTDTransducerTestGain(AUTDGainPtr *gain, int32_t idx, int32_t amp, int32_t phase) {
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
void AUTDSawModulation(AUTDModulationPtr *mod, int32_t freq) {
  auto *m = autd::SawModulation::Create(freq);
  *mod = m;
}
void AUTDSineModulation(AUTDModulationPtr *mod, int32_t freq, double amp, double offset) {
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

#pragma region Link
void AUTDSOEMLink(AUTDLinkPtr *out, const char *ifname, int32_t device_num) {
  auto *link = autd::SOEMLink::Create(std::string(ifname), device_num);
  *out = link;
}
void AUTDEtherCATLink(AUTDLinkPtr *out, const char *ipv4addr, const char *ams_net_id) {
  auto *link = autd::EthercatLink::Create(std::string(ipv4addr), std::string(ams_net_id));
  *out = link;
}
void AUTDLocalEtherCATLink(AUTDLinkPtr *out) {
  auto *link = autd::LocalEthercatLink::Create();
  *out = link;
}
void AUTDEmulatorLink(AUTDLinkPtr *out, const char *addr, int32_t port, AUTDControllerHandle handle) {
  auto *cnt = static_cast<autd::Controller *>(handle);
  auto *link = autd::EmulatorLink::Create(std::string(addr), port, cnt->geometry());
  *out = link;
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
int32_t AUTDDevIdForDeviceIdx(AUTDControllerHandle handle, int32_t dev_idx) {
  auto *cnt = static_cast<autd::Controller *>(handle);
  return cnt->geometry()->deviceIdForDeviceIdx(dev_idx);
}
int32_t AUTDDevIdForTransIdx(AUTDControllerHandle handle, int32_t trans_idx) {
  auto *cnt = static_cast<autd::Controller *>(handle);
  return cnt->geometry()->deviceIdForTransIdx(trans_idx);
}
double *AUTDTransPosition(AUTDControllerHandle handle, int32_t trans_idx) {
  auto *cnt = static_cast<autd::Controller *>(handle);
  auto pos = cnt->geometry()->position(trans_idx);
  auto *array = new double[3];
  array[0] = pos.x();
  array[1] = pos.y();
  array[2] = pos.z();
  return array;
}
double *AUTDTransDirection(AUTDControllerHandle handle, int32_t trans_idx) {
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
