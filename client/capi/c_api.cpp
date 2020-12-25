// File: c_api.cpp
// Project: capi
// Created Date: 02/07/2018
// Author: Shun Suzuki
// -----
// Last Modified: 25/12/2020
// Modified By: Shun Suzuki (suzuki@hapis.k.u-tokyo.ac.jp)
// -----
// Copyright (c) 2018-2020 Hapis Lab. All rights reserved.
//

#include <cerrno>
#include <cstdint>
#include <cstring>

#include "./autd3_c_api.h"
#include "autd3.hpp"
#include "emulator_link.hpp"
#include "soem_link.hpp"
#include "twincat_link.hpp"
#include "wrapper.hpp"

#pragma region Controller
void AUTDCreateController(void** out) {
  const auto ptr = autd::Controller::Create();
  auto* cnt = ControllerCreate(ptr);
  *out = cnt;
}
int32_t AUTDOpenControllerWith(void* const handle, void* const p_link) {
  auto* cnt = static_cast<ControllerWrapper*>(handle);
  auto* link = static_cast<LinkWrapper*>(p_link);
  cnt->ptr->OpenWith(move(link->ptr));
  LinkDelete(link);
  if (!cnt->ptr->is_open()) return ENXIO;
  return 0;
}
int32_t AUTDAddDevice(void* const handle, const double x, const double y, const double z, const double rz1, const double ry, const double rz2,
                      const int32_t group_id) {
  auto* cnt = static_cast<ControllerWrapper*>(handle);
  const auto res = cnt->ptr->geometry()->AddDevice(autd::Vector3(x, y, z), autd::Vector3(rz1, ry, rz2), group_id);
  return static_cast<int32_t>(res);
}
int32_t AUTDAddDeviceQuaternion(void* const handle, const double x, const double y, const double z, const double qua_w, const double qua_x,
                                const double qua_y, const double qua_z, const int32_t group_id) {
  auto* cnt = static_cast<ControllerWrapper*>(handle);
  const auto res = cnt->ptr->geometry()->AddDeviceQuaternion(autd::Vector3(x, y, z), autd::Quaternion(qua_w, qua_x, qua_y, qua_z), group_id);
  return static_cast<int32_t>(res);
}
bool AUTDCalibrate(void* const handle, int32_t smpl_freq, int32_t buf_size) {
  auto* cnt = static_cast<ControllerWrapper*>(handle);
  auto config = autd::Configuration::GetDefaultConfiguration();
  config.set_mod_sampling_freq(static_cast<autd::MOD_SAMPLING_FREQ>(smpl_freq));
  config.set_mod_buf_size(static_cast<autd::MOD_BUF_SIZE>(buf_size));
  return cnt->ptr->Calibrate(config);
}
void AUTDCloseController(void* const handle) {
  auto* cnt = static_cast<ControllerWrapper*>(handle);
  cnt->ptr->Close();
}
void AUTDClear(void* const handle) {
  auto* cnt = static_cast<ControllerWrapper*>(handle);
  cnt->ptr->Clear();
}
void AUTDFreeController(void* const handle) {
  auto* cnt = static_cast<ControllerWrapper*>(handle);
  ControllerDelete(cnt);
}
void AUTDSetSilentMode(void* const handle, const bool mode) {
  auto* cnt = static_cast<ControllerWrapper*>(handle);
  cnt->ptr->SetSilentMode(mode);
}
void AUTDStop(void* const handle) {
  auto* cnt = static_cast<ControllerWrapper*>(handle);
  cnt->ptr->Stop();
}
int32_t AUTDGetAdapterPointer(void** out) {
  size_t size;
  const auto adapters = autd::link::SOEMLink::EnumerateAdapters(&size);
  *out = EtherCATAdaptersCreate(adapters);
  return static_cast<int32_t>(size);
}
void AUTDGetAdapter(void* p_adapter, const int32_t index, char* desc, char* name) {
  auto* wrapper = static_cast<EtherCATAdaptersWrapper*>(p_adapter);
  const auto& desc_ = wrapper->adapters[index].first;
  const auto& name_ = wrapper->adapters[index].second;
  std::char_traits<char>::copy(desc, desc_.c_str(), desc_.size() + 1);
  std::char_traits<char>::copy(name, name_.c_str(), name_.size() + 1);
}
void AUTDFreeAdapterPointer(void* p_adapter) {
  auto* wrapper = static_cast<EtherCATAdaptersWrapper*>(p_adapter);
  EtherCATAdaptersDelete(wrapper);
}
int32_t AUTDGetFirmwareInfoListPointer(void* const handle, void** out) {
  auto* cnt = static_cast<ControllerWrapper*>(handle);
  const auto size = static_cast<int32_t>(cnt->ptr->geometry()->num_devices());
  auto* list = FirmwareInfoListCreate(cnt->ptr->firmware_info_list());
  *out = list;
  return size;
}
void AUTDGetFirmwareInfo(void* p_firm_info_list, const int32_t index, char* cpu_ver, char* fpga_ver) {
  auto* wrapper = static_cast<FirmwareInfoListWrapper*>(p_firm_info_list);
  const auto cpu_ver_ = wrapper->list[index].cpu_version();
  const auto fpga_ver_ = wrapper->list[index].fpga_version();
  std::char_traits<char>::copy(cpu_ver, cpu_ver_.c_str(), cpu_ver_.size() + 1);
  std::char_traits<char>::copy(fpga_ver, fpga_ver_.c_str(), fpga_ver_.size() + 1);
}
void AUTDFreeFirmwareInfoListPointer(void* p_firm_info_list) {
  auto* wrapper = static_cast<FirmwareInfoListWrapper*>(p_firm_info_list);
  FirmwareInfoListDelete(wrapper);
}
#pragma endregion

#pragma region Property
bool AUTDIsOpen(void* const handle) {
  auto* cnt = static_cast<ControllerWrapper*>(handle);
  return cnt->ptr->is_open();
}
bool AUTDIsSilentMode(void* const handle) {
  auto* cnt = static_cast<ControllerWrapper*>(handle);
  return cnt->ptr->silent_mode();
}
int32_t AUTDNumDevices(void* const handle) {
  auto* cnt = static_cast<ControllerWrapper*>(handle);
  const auto res = cnt->ptr->geometry()->num_devices();
  return static_cast<int32_t>(res);
}
int32_t AUTDNumTransducers(void* const handle) {
  auto* cnt = static_cast<ControllerWrapper*>(handle);
  const auto res = cnt->ptr->geometry()->num_transducers();
  return static_cast<int32_t>(res);
}
uint64_t AUTDRemainingInBuffer(void* const handle) {
  auto* cnt = static_cast<ControllerWrapper*>(handle);
  return static_cast<uint64_t>(cnt->ptr->remaining_in_buffer());
}
#pragma endregion

#pragma region Gain
void AUTDFocalPointGain(void** gain, const double x, const double y, const double z, const uint8_t duty) {
  auto* g = GainCreate(autd::gain::FocalPointGain::Create(autd::Vector3(x, y, z), duty));
  *gain = g;
}
void AUTDGroupedGain(void** gain, const int32_t* group_ids, void** gains, const int32_t size) {
  std::map<size_t, autd::GainPtr> gain_map;

  for (auto i = 0; i < size; i++) {
    const auto id = group_ids[i];
    auto* const gain_id = gains[i];
    auto* g = static_cast<GainWrapper*>(gain_id);
    gain_map[id] = g->ptr;
  }

  auto* g_gain = GainCreate(autd::gain::GroupedGain::Create(gain_map));

  *gain = g_gain;
}
void AUTDBesselBeamGain(void** gain, const double x, const double y, const double z, const double n_x, const double n_y, const double n_z,
                        const double theta_z, const uint8_t duty) {
  auto* g = GainCreate(autd::gain::BesselBeamGain::Create(autd::Vector3(x, y, z), autd::Vector3(n_x, n_y, n_z), theta_z, duty));
  *gain = g;
}
void AUTDPlaneWaveGain(void** gain, const double n_x, const double n_y, const double n_z, const uint8_t duty) {
  auto* g = GainCreate(autd::gain::PlaneWaveGain::Create(autd::Vector3(n_x, n_y, n_z), duty));
  *gain = g;
}
void AUTDCustomGain(void** gain, uint16_t* data, const int32_t data_length) {
  auto* g = GainCreate(autd::gain::CustomGain::Create(data, data_length));
  *gain = g;
}
void AUTDHoloGain(void** gain, double* points, double* amps, const int32_t size, int32_t method, void* params) {
  std::vector<autd::Vector3> holo;
  std::vector<double> amps_;
  for (auto i = 0; i < size; i++) {
    autd::Vector3 v(points[3 * i], points[3 * i + 1], points[3 * i + 2]);
    holo.push_back(v);
    amps_.push_back(amps[i]);
  }

  const auto method_ = static_cast<autd::gain::OPT_METHOD>(method);
  auto* g = GainCreate(autd::gain::HoloGain::Create(holo, amps_, method_, params));
  *gain = g;
}
void AUTDTransducerTestGain(void** gain, const int32_t idx, const uint8_t duty, const uint8_t phase) {
  auto* g = GainCreate(autd::gain::TransducerTestGain::Create(idx, duty, phase));
  *gain = g;
}
void AUTDNullGain(void** gain) {
  auto* g = GainCreate(autd::gain::NullGain::Create());
  *gain = g;
}
void AUTDDeleteGain(void* const gain) {
  auto* g = static_cast<GainWrapper*>(gain);
  GainDelete(g);
}
#pragma endregion

#pragma region Modulation
void AUTDModulation(void** mod, const uint8_t amp) {
  auto* m = ModulationCreate(autd::modulation::Modulation::Create(amp));
  *mod = m;
}
void AUTDCustomModulation(void** mod, uint8_t* buf, const uint32_t size) {
  auto* m = ModulationCreate(autd::modulation::Modulation::Create(0));
  m->ptr->buffer.resize(size, 0);
  std::memcpy(&m->ptr->buffer[0], buf, size);
  *mod = m;
}
void AUTDRawPCMModulation(void** mod, const char* filename, const double sampling_freq) {
  auto* m = ModulationCreate(autd::modulation::RawPCMModulation::Create(std::string(filename), sampling_freq));
  *mod = m;
}
void AUTDSquareModulation(void** mod, const int32_t freq, const uint8_t low, const uint8_t high) {
  auto* m = ModulationCreate(autd::modulation::SquareModulation::Create(freq, low, high));
  *mod = m;
}
void AUTDSawModulation(void** mod, const int32_t freq) {
  auto* m = ModulationCreate(autd::modulation::SawModulation::Create(freq));
  *mod = m;
}
void AUTDSineModulation(void** mod, const int32_t freq, const double amp, const double offset) {
  auto* m = ModulationCreate(autd::modulation::SineModulation::Create(freq, amp, offset));
  *mod = m;
}
void AUTDWavModulation(void** mod, const char* filename) {
  auto* m = ModulationCreate(autd::modulation::WavModulation::Create(std::string(filename)));
  *mod = m;
}
void AUTDDeleteModulation(void* const mod) {
  auto* m = static_cast<ModulationWrapper*>(mod);
  ModulationDelete(m);
}
#pragma endregion

#pragma region Sequence
void AUTDSequence(void** out) {
  auto* s = SequencePtrCreate(autd::sequence::PointSequence::Create());
  *out = s;
}
void AUTDSequenceAppendPoint(void* const seq, const double x, const double y, const double z) {
  auto* seq_w = static_cast<SequenceWrapper*>(seq);
  seq_w->ptr->AppendPoint(autd::Vector3(x, y, z));
}
void AUTDSequenceAppendPoints(void* const seq, double* points, const uint64_t size) {
  auto* seq_w = static_cast<SequenceWrapper*>(seq);
  std::vector<autd::Vector3> p;
  for (size_t i = 0; i < size; i++) {
    p.emplace_back(autd::Vector3(points[3 * i], points[3 * i + 1], points[3 * i + 2]));
  }
  seq_w->ptr->AppendPoints(p);
}
double AUTDSequenceSetFreq(void* const seq, const double freq) {
  auto* seq_w = static_cast<SequenceWrapper*>(seq);
  return seq_w->ptr->SetFrequency(freq);
}
double AUTDSequenceFreq(void* const seq) {
  auto* seq_w = static_cast<SequenceWrapper*>(seq);
  return seq_w->ptr->frequency();
}
double AUTDSequenceSamplingFreq(void* const seq) {
  auto* seq_w = static_cast<SequenceWrapper*>(seq);
  return seq_w->ptr->sampling_frequency();
}
uint16_t AUTDSequenceSamplingFreqDiv(void* const seq) {
  auto* seq_w = static_cast<SequenceWrapper*>(seq);
  return seq_w->ptr->sampling_frequency_division();
}
void AUTDCircumSequence(void** out, const double x, const double y, const double z, const double nx, const double ny, const double nz,
                        const double radius, const uint64_t n) {
  auto* s = SequencePtrCreate(autd::sequence::CircumSeq::Create(autd::Vector3(x, y, z), autd::Vector3(nx, ny, nz), radius, n));
  *out = s;
}
void AUTDDeleteSequence(void* const seq) {
  auto* seq_w = static_cast<SequenceWrapper*>(seq);
  SequenceDelete(seq_w);
}
#pragma endredion

#pragma region Link
void AUTDSOEMLink(void** out, const char* ifname, const int32_t device_num) {
  auto* link = LinkCreate(autd::link::SOEMLink::Create(std::string(ifname), device_num));
  *out = link;
}
void AUTDTwinCATLink(void** out, const char* ipv4addr, const char* ams_net_id) {
  auto* link = LinkCreate(autd::link::TwinCATLink::Create(std::string(ipv4addr), std::string(ams_net_id)));
  *out = link;
}
void AUTDLocalTwinCATLink(void** out) {
  auto* link = LinkCreate(autd::link::LocalTwinCATLink::Create());
  *out = link;
}
void AUTDEmulatorLink(void** out, const char* addr, const uint16_t port, void* const handle) {
  auto* cnt = static_cast<ControllerWrapper*>(handle);
  auto* link = LinkCreate(autd::link::EmulatorLink::Create(std::string(addr), port, cnt->ptr->geometry()));
  *out = link;
}
#pragma endregion

#pragma region LowLevelInterface
void AUTDAppendGain(void* const handle, void* const gain) {
  auto* cnt = static_cast<ControllerWrapper*>(handle);
  auto* g = static_cast<GainWrapper*>(gain);
  cnt->ptr->AppendGain(g->ptr);
}

void AUTDAppendGainSync(void* const handle, void* const gain, const bool wait_for_send) {
  auto* cnt = static_cast<ControllerWrapper*>(handle);
  auto* g = static_cast<GainWrapper*>(gain);
  cnt->ptr->AppendGainSync(g->ptr, wait_for_send);
}
void AUTDAppendModulation(void* const handle, void* const mod) {
  auto* cnt = static_cast<ControllerWrapper*>(handle);
  auto* m = static_cast<ModulationWrapper*>(mod);
  cnt->ptr->AppendModulation(m->ptr);
}
void AUTDAppendModulationSync(void* const handle, void* const mod) {
  auto* cnt = static_cast<ControllerWrapper*>(handle);
  auto* m = static_cast<ModulationWrapper*>(mod);
  cnt->ptr->AppendModulationSync(m->ptr);
}
void AUTDAppendSTMGain(void* const handle, void* const gain) {
  auto* cnt = static_cast<ControllerWrapper*>(handle);
  auto* g = static_cast<GainWrapper*>(gain);
  cnt->ptr->AppendSTMGain(g->ptr);
}
void AUTDStartSTModulation(void* const handle, const double freq) {
  auto* cnt = static_cast<ControllerWrapper*>(handle);
  cnt->ptr->StartSTModulation(freq);
}
void AUTDStopSTModulation(void* const handle) {
  auto* cnt = static_cast<ControllerWrapper*>(handle);
  cnt->ptr->StopSTModulation();
}
void AUTDFinishSTModulation(void* const handle) {
  auto* cnt = static_cast<ControllerWrapper*>(handle);
  cnt->ptr->FinishSTModulation();
}
void AUTDAppendSequence(void* const handle, void* const seq) {
  auto* cnt = static_cast<ControllerWrapper*>(handle);
  auto* s = static_cast<SequenceWrapper*>(seq);
  cnt->ptr->AppendSequence(s->ptr);
}
void AUTDFlush(void* const handle) {
  auto* cnt = static_cast<ControllerWrapper*>(handle);
  cnt->ptr->Flush();
}
int32_t AUTDDevIdxForTransIdx(void* const handle, const int32_t trans_idx) {
  auto* cnt = static_cast<ControllerWrapper*>(handle);
  const auto res = cnt->ptr->geometry()->device_idx_for_trans_idx(trans_idx);
  return static_cast<int32_t>(res);
}
double* AUTDTransPosition(void* const handle, const int32_t trans_idx) {
  auto* cnt = static_cast<ControllerWrapper*>(handle);
  const auto pos = cnt->ptr->geometry()->position(trans_idx);
  auto* array = new double[3];
  array[0] = pos.x();
  array[1] = pos.y();
  array[2] = pos.z();
  return array;
}
double* AUTDTransDirection(void* const handle, const int32_t trans_idx) {
  auto* cnt = static_cast<ControllerWrapper*>(handle);
  const auto dir = cnt->ptr->geometry()->direction(trans_idx);
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

void SetDebugLog(const DebugLogFunc func) { _debugLogFunc = func; }
#endif

#pragma endregion
