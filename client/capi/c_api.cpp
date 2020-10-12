// File: c_api.cpp
// Project: capi
// Created Date: 02/07/2018
// Author: Shun Suzuki
// -----
// Last Modified: 12/10/2020
// Modified By: Shun Suzuki (suzuki@hapis.k.u-tokyo.ac.jp)
// -----
// Copyright (c) 2018-2020 Hapis Lab. All rights reserved.
//

#include <errno.h>

#include <cstdint>

#include "./autd3_c_api.h"
#include "autd3.hpp"
#include "emulator_link.hpp"
#include "soem_link.hpp"
#include "twincat_link.hpp"
#include "wrapper.hpp"

#pragma region Controller
void AUTDCreateController(AUTDControllerHandle* out, int32_t version) {
  auto ptr = autd::Controller::Create(static_cast<autd::AUTD_VERSION>(version));
  auto* cnt = ControllerCreate(ptr);
  *out = cnt;
}
int32_t AUTDOpenControllerWith(AUTDControllerHandle handle, AUTDLinkPtr plink) {
  auto* cnt = static_cast<ControllerWrapper*>(handle);
  auto* link = static_cast<LinkWrapper*>(plink);
  cnt->ptr->OpenWith(link->ptr);
  if (!cnt->ptr->is_open()) return ENXIO;
  return 0;
}
int32_t AUTDAddDevice(AUTDControllerHandle handle, double x, double y, double z, double rz1, double ry, double rz2, int32_t group_id) {
  auto* cnt = static_cast<ControllerWrapper*>(handle);
  return cnt->ptr->geometry()->AddDevice(autd::Vector3(x, y, z), autd::Vector3(rz1, ry, rz2), group_id);
}
int32_t AUTDAddDeviceQuaternion(AUTDControllerHandle handle, double x, double y, double z, double qua_w, double qua_x, double qua_y, double qua_z,
                                int32_t group_id) {
  auto* cnt = static_cast<ControllerWrapper*>(handle);
  return cnt->ptr->geometry()->AddDeviceQuaternion(autd::Vector3(x, y, z), autd::Quaternion(qua_w, qua_x, qua_y, qua_z), group_id);
}
void AUTDDelDevice(AUTDControllerHandle handle, int32_t devId) {
  auto* cnt = static_cast<ControllerWrapper*>(handle);
  cnt->ptr->geometry()->DelDevice(devId);
}
void AUTDCalibrate(AUTDControllerHandle handle) {
  auto* cnt = static_cast<ControllerWrapper*>(handle);
  cnt->ptr->Calibrate();
}
void AUTDCloseController(AUTDControllerHandle handle) {
  auto* cnt = static_cast<ControllerWrapper*>(handle);
  cnt->ptr->Close();
}
void AUTDClear(AUTDControllerHandle handle) {
  auto* cnt = static_cast<ControllerWrapper*>(handle);
  cnt->ptr->Clear();
}
void AUTDFreeController(AUTDControllerHandle handle) {
  auto* cnt = static_cast<ControllerWrapper*>(handle);
  ControllerDelete(cnt);
}
void AUTDSetSilentMode(AUTDControllerHandle handle, bool mode) {
  auto* cnt = static_cast<ControllerWrapper*>(handle);
  cnt->ptr->SetSilentMode(mode);
}
void AUTDStop(AUTDControllerHandle handle) {
  auto* cnt = static_cast<ControllerWrapper*>(handle);
  cnt->ptr->Stop();
}
int32_t AUTDGetAdapterPointer(void** out) {
  int32_t size;
  auto adapters = autd::link::SOEMLink::EnumerateAdapters(&size);
  *out = EtherCATAdaptersCreate(adapters);
  return size;
}
void AUTDGetAdapter(void* p_adapter, int32_t index, char* desc, char* name) {
  auto* wrapper = static_cast<EtherCATAdaptersWrapper*>(p_adapter);
  auto desc_ = wrapper->adapters[index].first;
  auto name_ = wrapper->adapters[index].second;
  std::char_traits<char>::copy(desc, desc_.c_str(), desc_.size() + 1);
  std::char_traits<char>::copy(name, name_.c_str(), name_.size() + 1);
}
void AUTDFreeAdapterPointer(void* p_adapter) {
  auto* wrapper = static_cast<EtherCATAdaptersWrapper*>(p_adapter);
  EtherCATAdaptersDelete(wrapper);
}
int32_t AUTDGetFirmwareInfoListPointer(AUTDControllerHandle handle, void** out) {
  auto* cnt = static_cast<ControllerWrapper*>(handle);
  int32_t size = cnt->ptr->geometry()->numDevices();
  auto* list = FirmwareInfoListCreate(cnt->ptr->firmware_info_list());
  *out = list;
  return size;
}
void AUTDGetFirmwareInfo(void* pfirminfolist, int32_t index, char* cpu_ver, char* fpga_ver) {
  auto* wrapper = static_cast<FirmwareInfoListWrapper*>(pfirminfolist);
  auto cpu_ver_ = wrapper->list[index].cpu_version();
  auto fpga_ver_ = wrapper->list[index].fpga_version();
  std::char_traits<char>::copy(cpu_ver, cpu_ver_.c_str(), cpu_ver_.size() + 1);
  std::char_traits<char>::copy(fpga_ver, fpga_ver_.c_str(), fpga_ver_.size() + 1);
}
void AUTDFreeFirmwareInfoListPointer(void* pfirminfolist) {
  auto* wrapper = static_cast<FirmwareInfoListWrapper*>(pfirminfolist);
  FirmwareInfoListDelete(wrapper);
}
#pragma endregion

#pragma region Property
bool AUTDIsOpen(AUTDControllerHandle handle) {
  auto* cnt = static_cast<ControllerWrapper*>(handle);
  return cnt->ptr->is_open();
}
bool AUTDIsSilentMode(AUTDControllerHandle handle) {
  auto* cnt = static_cast<ControllerWrapper*>(handle);
  return cnt->ptr->silent_mode();
}
int32_t AUTDNumDevices(AUTDControllerHandle handle) {
  auto* cnt = static_cast<ControllerWrapper*>(handle);
  return cnt->ptr->geometry()->numDevices();
}
int32_t AUTDNumTransducers(AUTDControllerHandle handle) {
  auto* cnt = static_cast<ControllerWrapper*>(handle);
  return cnt->ptr->geometry()->numTransducers();
}
uint64_t AUTDRemainingInBuffer(AUTDControllerHandle handle) {
  auto* cnt = static_cast<ControllerWrapper*>(handle);
  return static_cast<uint64_t>(cnt->ptr->remaining_in_buffer());
}
#pragma endregion

#pragma region Gain
void AUTDFocalPointGain(AUTDGainPtr* gain, double x, double y, double z, uint8_t amp) {
  auto* g = GainCreate(autd::gain::FocalPointGain::Create(autd::Vector3(x, y, z), amp));
  *gain = g;
}
void AUTDGroupedGain(AUTDGainPtr* gain, int32_t* group_ids, AUTDGainPtr* gains, int32_t size) {
  std::map<int, autd::GainPtr> gainmap;

  for (int32_t i = 0; i < size; i++) {
    auto id = group_ids[i];
    auto gain_id = gains[i];
    auto* g = static_cast<GainWrapper*>(gain_id);
    gainmap[id] = g->ptr;
  }

  auto* ggain = GainCreate(autd::gain::GroupedGain::Create(gainmap));

  *gain = ggain;
}
void AUTDBesselBeamGain(AUTDGainPtr* gain, double x, double y, double z, double n_x, double n_y, double n_z, double theta_z) {
  auto* g = GainCreate(autd::gain::BesselBeamGain::Create(autd::Vector3(x, y, z), autd::Vector3(n_x, n_y, n_z), theta_z));
  *gain = g;
}
void AUTDPlaneWaveGain(AUTDGainPtr* gain, double n_x, double n_y, double n_z) {
  auto* g = GainCreate(autd::gain::PlaneWaveGain::Create(autd::Vector3(n_x, n_y, n_z)));
  *gain = g;
}
void AUTDCustomGain(AUTDGainPtr* gain, uint16_t* data, int32_t data_length) {
  auto* g = GainCreate(autd::gain::CustomGain::Create(data, data_length));
  *gain = g;
}
void AUTDHoloGain(AUTDGainPtr* gain, double* points, double* amps, int32_t size) {
  std::vector<autd::Vector3> holo;
  std::vector<double> amps_;
  for (int32_t i = 0; i < size; i++) {
    autd::Vector3 v(points[3 * i], points[3 * i + 1], points[3 * i + 2]);
    holo.push_back(v);
    amps_.push_back(amps[i]);
  }

  auto* g = GainCreate(autd::gain::HoloGainSdp::Create(holo, amps_));
  *gain = g;
}
void AUTDTransducerTestGain(AUTDGainPtr* gain, int32_t idx, int32_t amp, int32_t phase) {
  auto* g = GainCreate(autd::gain::TransducerTestGain::Create(idx, amp, phase));
  *gain = g;
}
void AUTDNullGain(AUTDGainPtr* gain) {
  auto* g = GainCreate(autd::gain::NullGain::Create());
  *gain = g;
}
void AUTDDeleteGain(AUTDGainPtr gain) {
  auto* g = static_cast<GainWrapper*>(gain);
  GainDelete(g);
}
#pragma endregion

#pragma region Modulation
void AUTDModulation(AUTDModulationPtr* mod, uint8_t amp) {
  auto* m = ModulationCreate(autd::modulation::Modulation::Create(amp));
  *mod = m;
}
void AUTDRawPCMModulation(AUTDModulationPtr* mod, const char* filename, double samp_freq) {
  auto* m = ModulationCreate(autd::modulation::RawPCMModulation::Create(std::string(filename), samp_freq));
  *mod = m;
}
void AUTDSawModulation(AUTDModulationPtr* mod, int32_t freq) {
  auto* m = ModulationCreate(autd::modulation::SawModulation::Create(freq));
  *mod = m;
}
void AUTDSineModulation(AUTDModulationPtr* mod, int32_t freq, double amp, double offset) {
  auto* m = ModulationCreate(autd::modulation::SineModulation::Create(freq, amp, offset));
  *mod = m;
}
void AUTDWavModulation(AUTDModulationPtr* mod, const char* filename) {
  auto* m = ModulationCreate(autd::modulation::WavModulation::Create(std::string(filename)));
  *mod = m;
}
void AUTDDeleteModulation(AUTDModulationPtr mod) {
  auto* m = static_cast<ModulationWrapper*>(mod);
  ModulationDelete(m);
}
#pragma endregion

#pragma region Sequence
void AUTDSequence(AUTDSequencePtr* out) {
  auto* s = SequencePtrCreate(autd::sequence::PointSequence::Create());
  *out = s;
}
void AUTDSequenceAppnedPoint(AUTDSequencePtr handle, double x, double y, double z) {
  auto* seq = static_cast<SequenceWrapper*>(handle);
  seq->ptr->AppendPoint(autd::Vector3(x, y, z));
}
void AUTDSequenceAppnedPoints(AUTDSequencePtr handle, double* points, uint64_t size) {
  auto* seq = static_cast<SequenceWrapper*>(handle);
  std::vector<autd::Vector3> p;
  for (size_t i = 0; i < size; i++) {
    p.push_back(autd::Vector3(points[3 * i], points[3 * i + 1], points[3 * i + 2]));
  }
  seq->ptr->AppendPoints(p);
}
double AUTDSequenceSetFreq(AUTDSequencePtr handle, double freq) {
  auto* seq = static_cast<SequenceWrapper*>(handle);
  return seq->ptr->SetFrequency(freq);
}
double AUTDSequenceFreq(AUTDSequencePtr handle) {
  auto* seq = static_cast<SequenceWrapper*>(handle);
  return seq->ptr->frequency();
}
double AUTDSequenceSamplingFreq(AUTDSequencePtr handle) {
  auto* seq = static_cast<SequenceWrapper*>(handle);
  return seq->ptr->sampling_frequency();
}
uint16_t AUTDSequenceSamplingFreqDiv(AUTDSequencePtr handle) {
  auto* seq = static_cast<SequenceWrapper*>(handle);
  return seq->ptr->sampling_frequency_division();
}
void AUTDCircumSequence(AUTDSequencePtr* out, double x, double y, double z, double nx, double ny, double nz, double radius, uint64_t n) {
  auto* s = SequencePtrCreate(autd::sequence::CircumSeq::Create(autd::Vector3(x, y, z), autd::Vector3(nx, ny, nz), radius, n));
  *out = s;
}
void AUTDDeleteSequence(AUTDSequencePtr handle) {
  auto* seq = static_cast<SequenceWrapper*>(handle);
  SequenceDelete(seq);
}
#pragma endredion

#pragma region Link
void AUTDSOEMLink(AUTDLinkPtr* out, const char* ifname, int32_t device_num) {
  auto* link = LinkCreate(autd::link::SOEMLink::Create(std::string(ifname), device_num));
  *out = link;
}
void AUTDTwinCATLink(AUTDLinkPtr* out, const char* ipv4addr, const char* ams_net_id) {
  auto* link = LinkCreate(autd::link::TwinCATLink::Create(std::string(ipv4addr), std::string(ams_net_id)));
  *out = link;
}
void AUTDLocalTwinCATLink(AUTDLinkPtr* out) {
  auto* link = LinkCreate(autd::link::LocalTwinCATLink::Create());
  *out = link;
}
void AUTDEmulatorLink(AUTDLinkPtr* out, const char* addr, int32_t port, AUTDControllerHandle handle) {
  auto* cnt = static_cast<ControllerWrapper*>(handle);
  auto* link = LinkCreate(autd::link::EmulatorLink::Create(std::string(addr), port, cnt->ptr->geometry()));
  *out = link;
}
#pragma endregion

#pragma region LowLevelInterface
void AUTDAppendGain(AUTDControllerHandle handle, AUTDGainPtr gain) {
  auto* cnt = static_cast<ControllerWrapper*>(handle);
  auto* g = static_cast<GainWrapper*>(gain);
  cnt->ptr->AppendGain(g->ptr);
}

void AUTDAppendGainSync(AUTDControllerHandle handle, AUTDGainPtr gain, bool wait_for_send) {
  auto* cnt = static_cast<ControllerWrapper*>(handle);
  auto* g = static_cast<GainWrapper*>(gain);
  cnt->ptr->AppendGainSync(g->ptr);
}
void AUTDAppendModulation(AUTDControllerHandle handle, AUTDModulationPtr mod) {
  auto* cnt = static_cast<ControllerWrapper*>(handle);
  auto* m = static_cast<ModulationWrapper*>(mod);
  cnt->ptr->AppendModulation(m->ptr);
}
void AUTDAppendModulationSync(AUTDControllerHandle handle, AUTDModulationPtr mod) {
  auto* cnt = static_cast<ControllerWrapper*>(handle);
  auto* m = static_cast<ModulationWrapper*>(mod);
  cnt->ptr->AppendModulationSync(m->ptr);
}
void AUTDAppendSTMGain(AUTDControllerHandle handle, AUTDGainPtr gain) {
  auto* cnt = static_cast<ControllerWrapper*>(handle);
  auto* g = static_cast<GainWrapper*>(gain);
  cnt->ptr->AppendSTMGain(g->ptr);
}
void AUTDStartSTModulation(AUTDControllerHandle handle, double freq) {
  auto* cnt = static_cast<ControllerWrapper*>(handle);
  cnt->ptr->StartSTModulation(freq);
}
void AUTDStopSTModulation(AUTDControllerHandle handle) {
  auto* cnt = static_cast<ControllerWrapper*>(handle);
  cnt->ptr->StopSTModulation();
}
void AUTDFinishSTModulation(AUTDControllerHandle handle) {
  auto* cnt = static_cast<ControllerWrapper*>(handle);
  cnt->ptr->FinishSTModulation();
}
void AUTDAppendSequence(AUTDControllerHandle handle, AUTDSequencePtr seq) {
  auto* cnt = static_cast<ControllerWrapper*>(handle);
  auto* s = static_cast<SequenceWrapper*>(seq);
  cnt->ptr->AppendSequence(s->ptr);
}
void AUTDFlush(AUTDControllerHandle handle) {
  auto* cnt = static_cast<ControllerWrapper*>(handle);
  cnt->ptr->Flush();
}
int32_t AUTDDevIdForDeviceIdx(AUTDControllerHandle handle, int32_t dev_idx) {
  auto* cnt = static_cast<ControllerWrapper*>(handle);
  return cnt->ptr->geometry()->deviceIdForDeviceIdx(dev_idx);
}
int32_t AUTDDevIdForTransIdx(AUTDControllerHandle handle, int32_t trans_idx) {
  auto* cnt = static_cast<ControllerWrapper*>(handle);
  return cnt->ptr->geometry()->deviceIdForTransIdx(trans_idx);
}
double* AUTDTransPosition(AUTDControllerHandle handle, int32_t trans_idx) {
  auto* cnt = static_cast<ControllerWrapper*>(handle);
  auto pos = cnt->ptr->geometry()->position(trans_idx);
  auto* array = new double[3];
  array[0] = pos.x();
  array[1] = pos.y();
  array[2] = pos.z();
  return array;
}
double* AUTDTransDirection(AUTDControllerHandle handle, int32_t trans_idx) {
  auto* cnt = static_cast<ControllerWrapper*>(handle);
  auto dir = cnt->ptr->geometry()->direction(trans_idx);
  auto* array = new double[3];
  array[0] = dir.x();
  array[1] = dir.y();
  array[2] = dir.z();
  return array;
}
#pragma endregion

#pragma region Debug
#ifdef UNITY_DEBUG
DebugLogFunc _debugLogFunc = nullptr;

void DebugLog(const char* msg) {
  if (_debugLogFunc != nullptr) _debugLogFunc(msg);
}

void SetDebugLog(DebugLogFunc func) { _debugLogFunc = func; }
#endif

#pragma endregion
