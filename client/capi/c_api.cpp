// File: c_api.cpp
// Project: capi
// Created Date: 02/07/2018
// Author: Shun Suzuki
// -----
// Last Modified: 28/12/2020
// Modified By: Shun Suzuki (suzuki@hapis.k.u-tokyo.ac.jp)
// -----
// Copyright (c) 2018-2020 Hapis Lab. All rights reserved.
//

#include <cerrno>
#include <cstdint>
#include <cstring>
#include <utility>

#include "./autd3_c_api.h"
#include "autd3.hpp"
#include "emulator_link.hpp"
#include "soem_link.hpp"
#include "twincat_link.hpp"
#include "wrapper.hpp"

#pragma region Controller
void AUTDCreateController(VOID_PTR* out) {
  auto ptr = autd::Controller::Create();
  auto* cnt = ControllerCreate(std::move(ptr));
  *out = cnt;
}
int32_t AUTDOpenControllerWith(VOID_PTR const handle, VOID_PTR const p_link) {
  auto* cnt = static_cast<ControllerWrapper*>(handle);
  auto* link = static_cast<LinkWrapper*>(p_link);
  cnt->ptr->OpenWith(move(link->ptr));
  LinkDelete(link);
  if (!cnt->ptr->is_open()) return ENXIO;
  return 0;
}
int32_t AUTDAddDevice(VOID_PTR const handle, const float x, const float y, const float z, const float rz1, const float ry, const float rz2,
                      const int32_t group_id) {
  auto* cnt = static_cast<ControllerWrapper*>(handle);
  const auto res = cnt->ptr->geometry()->AddDevice(autd::Vector3(x, y, z), autd::Vector3(rz1, ry, rz2), group_id);
  return static_cast<int32_t>(res);
}
int32_t AUTDAddDeviceQuaternion(VOID_PTR const handle, const float x, const float y, const float z, const float qua_w, const float qua_x,
                                const float qua_y, const float qua_z, const int32_t group_id) {
  auto* cnt = static_cast<ControllerWrapper*>(handle);
  const auto res = cnt->ptr->geometry()->AddDeviceQuaternion(autd::Vector3(x, y, z), autd::Quaternion(qua_w, qua_x, qua_y, qua_z), group_id);
  return static_cast<int32_t>(res);
}
bool AUTDCalibrate(VOID_PTR const handle, int32_t smpl_freq, int32_t buf_size) {
  auto* cnt = static_cast<ControllerWrapper*>(handle);
  auto config = autd::Configuration::GetDefaultConfiguration();
  config.set_mod_sampling_freq(static_cast<autd::MOD_SAMPLING_FREQ>(smpl_freq));
  config.set_mod_buf_size(static_cast<autd::MOD_BUF_SIZE>(buf_size));
  return cnt->ptr->Calibrate(config);
}
void AUTDCloseController(VOID_PTR const handle) {
  auto* cnt = static_cast<ControllerWrapper*>(handle);
  cnt->ptr->Close();
}
void AUTDClear(VOID_PTR const handle) {
  auto* cnt = static_cast<ControllerWrapper*>(handle);
  cnt->ptr->Clear();
}
void AUTDFreeController(VOID_PTR const handle) {
  auto* cnt = static_cast<ControllerWrapper*>(handle);
  ControllerDelete(cnt);
}
void AUTDSetSilentMode(VOID_PTR const handle, const bool mode) {
  auto* cnt = static_cast<ControllerWrapper*>(handle);
  cnt->ptr->SetSilentMode(mode);
}
void AUTDStop(VOID_PTR const handle) {
  auto* cnt = static_cast<ControllerWrapper*>(handle);
  cnt->ptr->Stop();
}
int32_t AUTDGetAdapterPointer(VOID_PTR* out) {
  size_t size;
  const auto adapters = autd::link::SOEMLink::EnumerateAdapters(&size);
  *out = EtherCATAdaptersCreate(adapters);
  return static_cast<int32_t>(size);
}
void AUTDGetAdapter(VOID_PTR p_adapter, const int32_t index, char* desc, char* name) {
  auto* wrapper = static_cast<EtherCATAdaptersWrapper*>(p_adapter);
  const auto& desc_ = wrapper->adapters[index].first;
  const auto& name_ = wrapper->adapters[index].second;
  std::char_traits<char>::copy(desc, desc_.c_str(), desc_.size() + 1);
  std::char_traits<char>::copy(name, name_.c_str(), name_.size() + 1);
}
void AUTDFreeAdapterPointer(VOID_PTR p_adapter) {
  auto* wrapper = static_cast<EtherCATAdaptersWrapper*>(p_adapter);
  EtherCATAdaptersDelete(wrapper);
}
int32_t AUTDGetFirmwareInfoListPointer(VOID_PTR const handle, VOID_PTR* out) {
  auto* cnt = static_cast<ControllerWrapper*>(handle);
  const auto size = static_cast<int32_t>(cnt->ptr->geometry()->num_devices());
  auto* list = FirmwareInfoListCreate(cnt->ptr->firmware_info_list());
  *out = list;
  return size;
}
void AUTDGetFirmwareInfo(VOID_PTR p_firm_info_list, const int32_t index, char* cpu_ver, char* fpga_ver) {
  auto* wrapper = static_cast<FirmwareInfoListWrapper*>(p_firm_info_list);
  const auto cpu_ver_ = wrapper->list[index].cpu_version();
  const auto fpga_ver_ = wrapper->list[index].fpga_version();
  std::char_traits<char>::copy(cpu_ver, cpu_ver_.c_str(), cpu_ver_.size() + 1);
  std::char_traits<char>::copy(fpga_ver, fpga_ver_.c_str(), fpga_ver_.size() + 1);
}
void AUTDFreeFirmwareInfoListPointer(VOID_PTR p_firm_info_list) {
  auto* wrapper = static_cast<FirmwareInfoListWrapper*>(p_firm_info_list);
  FirmwareInfoListDelete(wrapper);
}
#pragma endregion

#pragma region Property
bool AUTDIsOpen(VOID_PTR const handle) {
  auto* cnt = static_cast<ControllerWrapper*>(handle);
  return cnt->ptr->is_open();
}
bool AUTDIsSilentMode(VOID_PTR const handle) {
  auto* cnt = static_cast<ControllerWrapper*>(handle);
  return cnt->ptr->silent_mode();
}
float AUTDWavelength(VOID_PTR const handle) {
  auto* cnt = static_cast<ControllerWrapper*>(handle);
  return cnt->ptr->geometry()->wavelength();
}
void AUTDSetWavelength(VOID_PTR const handle, const float wavelength) {
  auto* cnt = static_cast<ControllerWrapper*>(handle);
  return cnt->ptr->geometry()->set_wavelength(wavelength);
}
void AUTDSetDelay(VOID_PTR handle, const uint16_t* delay, const int32_t data_length) {
  auto* cnt = static_cast<ControllerWrapper*>(handle);

  const int32_t dev_num = data_length / autd::NUM_TRANS_IN_UNIT;
  std::vector<autd::AUTDDataArray> delay_(dev_num);
  auto dev_idx = 0;
  auto tran_idx = 0;
  for (auto i = 0; i < data_length; i++) {
    delay_[dev_idx][tran_idx++] = delay[i];
    if (tran_idx == autd::NUM_TRANS_IN_UNIT) {
      dev_idx++;
      tran_idx = 0;
    }
  }
  cnt->ptr->SetDelay(delay_);
}
int32_t AUTDNumDevices(VOID_PTR const handle) {
  auto* cnt = static_cast<ControllerWrapper*>(handle);
  const auto res = cnt->ptr->geometry()->num_devices();
  return static_cast<int32_t>(res);
}
int32_t AUTDNumTransducers(VOID_PTR const handle) {
  auto* cnt = static_cast<ControllerWrapper*>(handle);
  const auto res = cnt->ptr->geometry()->num_transducers();
  return static_cast<int32_t>(res);
}
uint64_t AUTDRemainingInBuffer(VOID_PTR const handle) {
  auto* cnt = static_cast<ControllerWrapper*>(handle);
  return static_cast<uint64_t>(cnt->ptr->remaining_in_buffer());
}
#pragma endregion

#pragma region Gain
void AUTDFocalPointGain(VOID_PTR* gain, const float x, const float y, const float z, const uint8_t duty) {
  auto* g = GainCreate(autd::gain::FocalPointGain::Create(autd::Vector3(x, y, z), duty));
  *gain = g;
}
void AUTDGroupedGain(VOID_PTR* gain, const int32_t* group_ids, VOID_PTR const* in_gains, const int32_t size) {
  std::map<size_t, autd::GainPtr> gain_map;

  for (auto i = 0; i < size; i++) {
    const auto id = group_ids[i];
    auto* const gain_id = in_gains[i];
    auto* g = static_cast<GainWrapper*>(gain_id);
    gain_map[id] = g->ptr;
  }

  auto* g_gain = GainCreate(autd::gain::GroupedGain::Create(gain_map));

  *gain = g_gain;
}
void AUTDBesselBeamGain(VOID_PTR* gain, const float x, const float y, const float z, const float n_x, const float n_y, const float n_z,
                        const float theta_z, const uint8_t duty) {
  auto* g = GainCreate(autd::gain::BesselBeamGain::Create(autd::Vector3(x, y, z), autd::Vector3(n_x, n_y, n_z), theta_z, duty));
  *gain = g;
}
void AUTDPlaneWaveGain(VOID_PTR* gain, const float n_x, const float n_y, const float n_z, const uint8_t duty) {
  auto* g = GainCreate(autd::gain::PlaneWaveGain::Create(autd::Vector3(n_x, n_y, n_z), duty));
  *gain = g;
}
void AUTDCustomGain(VOID_PTR* gain, const uint16_t* data, const int32_t data_length) {
  auto* g = GainCreate(autd::gain::CustomGain::Create(data, data_length));
  *gain = g;
}
void AUTDHoloGain(VOID_PTR* gain, const float* points, const float* amps, const int32_t size, int32_t method, VOID_PTR params) {
  std::vector<autd::Vector3> holo;
  std::vector<float> amps_;
  for (auto i = 0; i < size; i++) {
    autd::Vector3 v(points[3 * i], points[3 * i + 1], points[3 * i + 2]);
    holo.emplace_back(v);
    amps_.emplace_back(amps[i]);
  }

  const auto method_ = static_cast<autd::gain::OPT_METHOD>(method);
  auto* g = GainCreate(autd::gain::HoloGain::Create(holo, amps_, method_, params));
  *gain = g;
}
void AUTDTransducerTestGain(VOID_PTR* gain, const int32_t idx, const uint8_t duty, const uint8_t phase) {
  auto* g = GainCreate(autd::gain::TransducerTestGain::Create(idx, duty, phase));
  *gain = g;
}
void AUTDNullGain(VOID_PTR* gain) {
  auto* g = GainCreate(autd::gain::NullGain::Create());
  *gain = g;
}
void AUTDDeleteGain(VOID_PTR const gain) {
  auto* g = static_cast<GainWrapper*>(gain);
  GainDelete(g);
}
#pragma endregion

#pragma region Modulation
void AUTDModulation(VOID_PTR* mod, const uint8_t amp) {
  auto* m = ModulationCreate(autd::modulation::Modulation::Create(amp));
  *mod = m;
}
void AUTDCustomModulation(VOID_PTR* mod, const uint8_t* buf, const uint32_t size) {
  auto* m = ModulationCreate(autd::modulation::Modulation::Create(0));
  m->ptr->buffer.resize(size, 0);
  std::memcpy(&m->ptr->buffer[0], buf, size);
  *mod = m;
}
void AUTDRawPCMModulation(VOID_PTR* mod, const char* filename, const float sampling_freq) {
  auto* m = ModulationCreate(autd::modulation::RawPCMModulation::Create(std::string(filename), sampling_freq));
  *mod = m;
}
void AUTDSquareModulation(VOID_PTR* mod, const int32_t freq, const uint8_t low, const uint8_t high) {
  auto* m = ModulationCreate(autd::modulation::SquareModulation::Create(freq, low, high));
  *mod = m;
}
void AUTDSawModulation(VOID_PTR* mod, const int32_t freq) {
  auto* m = ModulationCreate(autd::modulation::SawModulation::Create(freq));
  *mod = m;
}
void AUTDSineModulation(VOID_PTR* mod, const int32_t freq, const float amp, const float offset) {
  auto* m = ModulationCreate(autd::modulation::SineModulation::Create(freq, amp, offset));
  *mod = m;
}
void AUTDWavModulation(VOID_PTR* mod, const char* filename) {
  auto* m = ModulationCreate(autd::modulation::WavModulation::Create(std::string(filename)));
  *mod = m;
}
void AUTDDeleteModulation(VOID_PTR const mod) {
  auto* m = static_cast<ModulationWrapper*>(mod);
  ModulationDelete(m);
}
#pragma endregion

#pragma region Sequence
void AUTDSequence(VOID_PTR* out) {
  auto* s = SequencePtrCreate(autd::sequence::PointSequence::Create());
  *out = s;
}
void AUTDSequenceAppendPoint(VOID_PTR const seq, const float x, const float y, const float z) {
  auto* seq_w = static_cast<SequenceWrapper*>(seq);
  seq_w->ptr->AppendPoint(autd::Vector3(x, y, z));
}
void AUTDSequenceAppendPoints(VOID_PTR const seq, const float* points, const uint64_t size) {
  auto* seq_w = static_cast<SequenceWrapper*>(seq);
  std::vector<autd::Vector3> p;
  for (size_t i = 0; i < size; i++) {
    p.emplace_back(autd::Vector3(points[3 * i], points[3 * i + 1], points[3 * i + 2]));
  }
  seq_w->ptr->AppendPoints(p);
}
float AUTDSequenceSetFreq(VOID_PTR const seq, const float freq) {
  auto* seq_w = static_cast<SequenceWrapper*>(seq);
  return seq_w->ptr->SetFrequency(freq);
}
float AUTDSequenceFreq(VOID_PTR const seq) {
  auto* seq_w = static_cast<SequenceWrapper*>(seq);
  return seq_w->ptr->frequency();
}
float AUTDSequenceSamplingFreq(VOID_PTR const seq) {
  auto* seq_w = static_cast<SequenceWrapper*>(seq);
  return seq_w->ptr->sampling_frequency();
}
uint16_t AUTDSequenceSamplingFreqDiv(VOID_PTR const seq) {
  auto* seq_w = static_cast<SequenceWrapper*>(seq);
  return seq_w->ptr->sampling_frequency_division();
}
void AUTDCircumSequence(VOID_PTR* out, const float x, const float y, const float z, const float nx, const float ny, const float nz,
                        const float radius, const uint64_t n) {
  auto* s = SequencePtrCreate(autd::sequence::CircumSeq::Create(autd::Vector3(x, y, z), autd::Vector3(nx, ny, nz), radius, n));
  *out = s;
}
void AUTDDeleteSequence(VOID_PTR const seq) {
  auto* seq_w = static_cast<SequenceWrapper*>(seq);
  SequenceDelete(seq_w);
}
#pragma endredion

#pragma region Link
void AUTDSOEMLink(VOID_PTR* out, const char* ifname, const int32_t device_num) {
  auto* link = LinkCreate(autd::link::SOEMLink::Create(std::string(ifname), device_num));
  *out = link;
}
void AUTDTwinCATLink(VOID_PTR* out, const char* ipv4_addr, const char* ams_net_id) {
  auto* link = LinkCreate(autd::link::TwinCATLink::Create(std::string(ipv4_addr), std::string(ams_net_id)));
  *out = link;
}
void AUTDLocalTwinCATLink(VOID_PTR* out) {
  auto* link = LinkCreate(autd::link::LocalTwinCATLink::Create());
  *out = link;
}
void AUTDEmulatorLink(VOID_PTR* out, const char* addr, const uint16_t port, VOID_PTR const handle) {
  auto* cnt = static_cast<ControllerWrapper*>(handle);
  auto* link = LinkCreate(autd::link::EmulatorLink::Create(std::string(addr), port, cnt->ptr->geometry()));
  *out = link;
}
#pragma endregion

#pragma region LowLevelInterface
void AUTDAppendGain(VOID_PTR const handle, VOID_PTR const gain) {
  auto* cnt = static_cast<ControllerWrapper*>(handle);
  auto* g = static_cast<GainWrapper*>(gain);
  cnt->ptr->AppendGain(g->ptr);
}

void AUTDAppendGainSync(VOID_PTR const handle, VOID_PTR const gain, const bool wait_for_send) {
  auto* cnt = static_cast<ControllerWrapper*>(handle);
  auto* g = static_cast<GainWrapper*>(gain);
  cnt->ptr->AppendGainSync(g->ptr, wait_for_send);
}
void AUTDAppendModulation(VOID_PTR const handle, VOID_PTR const mod) {
  auto* cnt = static_cast<ControllerWrapper*>(handle);
  auto* m = static_cast<ModulationWrapper*>(mod);
  cnt->ptr->AppendModulation(m->ptr);
}
void AUTDAppendModulationSync(VOID_PTR const handle, VOID_PTR const mod) {
  auto* cnt = static_cast<ControllerWrapper*>(handle);
  auto* m = static_cast<ModulationWrapper*>(mod);
  cnt->ptr->AppendModulationSync(m->ptr);
}
void AUTDAppendSTMGain(VOID_PTR const handle, VOID_PTR const gain) {
  auto* cnt = static_cast<ControllerWrapper*>(handle);
  auto* g = static_cast<GainWrapper*>(gain);
  cnt->ptr->AppendSTMGain(g->ptr);
}
void AUTDStartSTModulation(VOID_PTR const handle, const float freq) {
  auto* cnt = static_cast<ControllerWrapper*>(handle);
  cnt->ptr->StartSTModulation(freq);
}
void AUTDStopSTModulation(VOID_PTR const handle) {
  auto* cnt = static_cast<ControllerWrapper*>(handle);
  cnt->ptr->StopSTModulation();
}
void AUTDFinishSTModulation(VOID_PTR const handle) {
  auto* cnt = static_cast<ControllerWrapper*>(handle);
  cnt->ptr->FinishSTModulation();
}
void AUTDAppendSequence(VOID_PTR const handle, VOID_PTR const seq) {
  auto* cnt = static_cast<ControllerWrapper*>(handle);
  auto* s = static_cast<SequenceWrapper*>(seq);
  cnt->ptr->AppendSequence(s->ptr);
}
void AUTDFlush(VOID_PTR const handle) {
  auto* cnt = static_cast<ControllerWrapper*>(handle);
  cnt->ptr->Flush();
}
int32_t AUTDDeviceIdxForTransIdx(VOID_PTR const handle, const int32_t global_trans_idx) {
  auto* cnt = static_cast<ControllerWrapper*>(handle);
  const auto res = cnt->ptr->geometry()->device_idx_for_trans_idx(global_trans_idx);
  return static_cast<int32_t>(res);
}
float* AUTDTransPositionByGlobal(VOID_PTR const handle, const int32_t global_trans_idx) {
  auto* cnt = static_cast<ControllerWrapper*>(handle);
  const auto pos = cnt->ptr->geometry()->position(global_trans_idx);
  auto* array = new float[3];
  array[0] = pos.x();
  array[1] = pos.y();
  array[2] = pos.z();
  return array;
}
float* AUTDTransPositionByLocal(VOID_PTR const handle, const int32_t device_idx, const int32_t local_trans_idx) {
  auto* cnt = static_cast<ControllerWrapper*>(handle);
  const auto pos = cnt->ptr->geometry()->position(device_idx, local_trans_idx);
  auto* array = new float[3];
  array[0] = pos.x();
  array[1] = pos.y();
  array[2] = pos.z();
  return array;
}
float* AUTDDeviceDirection(VOID_PTR const handle, const int32_t device_idx) {
  auto* cnt = static_cast<ControllerWrapper*>(handle);
  const auto dir = cnt->ptr->geometry()->direction(device_idx);
  auto* array = new float[3];
  array[0] = dir.x();
  array[1] = dir.y();
  array[2] = dir.z();
  return array;
}
#pragma endregion
