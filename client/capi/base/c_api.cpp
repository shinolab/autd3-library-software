// File: c_api.cpp
// Project: capi
// Created Date: 02/07/2018
// Author: Shun Suzuki
// -----
// Last Modified: 11/05/2021
// Modified By: Shun Suzuki (suzuki@hapis.k.u-tokyo.ac.jp)
// -----
// Copyright (c) 2018-2020 Hapis Lab. All rights reserved.
//

#include <cstdint>
#include <cstring>
#include <utility>

#include "./autd3_c_api.h"
#include "autd3.hpp"
#include "wrapper.hpp"
#include "wrapper_gain.hpp"
#include "wrapper_link.hpp"
#include "wrapper_modulation.hpp"

namespace {
std::string& LastError() {
  static std::string msg("");
  return msg;
}
}  // namespace

#pragma region Controller
void AUTDCreateController(void** out) {
  auto ptr = autd::Controller::Create();
  auto* cnt = ControllerCreate(std::move(ptr));
  *out = cnt;
}
bool AUTDOpenControllerWith(void* const handle, void* const p_link) {
  auto* cnt = static_cast<ControllerWrapper*>(handle);
  auto* link = static_cast<LinkWrapper*>(p_link);
  auto res = cnt->ptr->OpenWith(move(link->ptr));
  LinkDelete(link);
  if (res.is_err()) LastError() = res.unwrap_err();
  return res.unwrap_or(false);
}
int32_t AUTDAddDevice(void* const handle, const float x, const float y, const float z, const float rz1, const float ry, const float rz2,
                      const int32_t group_id) {
  auto* cnt = static_cast<ControllerWrapper*>(handle);
  const auto res =
      cnt->ptr->geometry()->AddDevice(autd::Vector3(static_cast<double>(x), static_cast<double>(y), static_cast<double>(z)),
                                      autd::Vector3(static_cast<double>(rz1), static_cast<double>(ry), static_cast<double>(rz2)), group_id);
  return static_cast<int32_t>(res);
}
int32_t AUTDAddDeviceQuaternion(void* const handle, const float x, const float y, const float z, const float qua_w, const float qua_x,
                                const float qua_y, const float qua_z, const int32_t group_id) {
  auto* cnt = static_cast<ControllerWrapper*>(handle);
  const auto res = cnt->ptr->geometry()->AddDeviceQuaternion(
      autd::Vector3(static_cast<double>(x), static_cast<double>(y), static_cast<double>(z)),
      autd::Quaternion(static_cast<double>(qua_w), static_cast<double>(qua_x), static_cast<double>(qua_y), static_cast<double>(qua_z)), group_id);
  return static_cast<int32_t>(res);
}
int32_t AUTDDeleteDevice(void* const handle, const int32_t idx) {
  auto* cnt = static_cast<ControllerWrapper*>(handle);
  const auto res = cnt->ptr->geometry()->DelDevice(static_cast<size_t>(idx));
  return static_cast<int32_t>(res);
}
void AUTDClearDevices(void* const handle) {
  auto* cnt = static_cast<ControllerWrapper*>(handle);
  cnt->ptr->geometry()->ClearDevices();
}
bool AUTDSynchronize(void* const handle, int32_t smpl_freq, int32_t buf_size) {
  auto* cnt = static_cast<ControllerWrapper*>(handle);
  auto config = autd::Configuration::GetDefaultConfiguration();
  config.set_mod_sampling_freq(static_cast<autd::MOD_SAMPLING_FREQ>(smpl_freq));
  config.set_mod_buf_size(static_cast<autd::MOD_BUF_SIZE>(buf_size));
  auto res = cnt->ptr->Synchronize(config);
  if (res.is_err()) LastError() = res.unwrap_err();
  return res.unwrap_or(false);
}
bool AUTDCloseController(void* const handle) {
  auto* cnt = static_cast<ControllerWrapper*>(handle);
  auto res = cnt->ptr->Close();
  if (res.is_err()) LastError() = res.unwrap_err();
  return res.unwrap_or(false);
}
bool AUTDClear(void* const handle) {
  auto* cnt = static_cast<ControllerWrapper*>(handle);
  auto res = cnt->ptr->Clear();
  if (res.is_err()) LastError() = res.unwrap_err();
  return res.unwrap_or(false);
}
void AUTDFreeController(void* const handle) {
  auto* cnt = static_cast<ControllerWrapper*>(handle);
  ControllerDelete(cnt);
}
void AUTDSetSilentMode(void* const handle, const bool mode) {
  auto* cnt = static_cast<ControllerWrapper*>(handle);
  cnt->ptr->SetSilentMode(mode);
}
bool AUTDStop(void* const handle) {
  auto* cnt = static_cast<ControllerWrapper*>(handle);
  auto res = cnt->ptr->Stop();
  if (res.is_err()) LastError() = res.unwrap_err();
  return res.unwrap_or(false);
}
int32_t AUTDGetFirmwareInfoListPointer(void* const handle, void** out) {
  auto* cnt = static_cast<ControllerWrapper*>(handle);
  const auto size = static_cast<int32_t>(cnt->ptr->geometry()->num_devices());
  auto res = cnt->ptr->firmware_info_list();
  if (res.is_err()) {
    LastError() = res.unwrap_err();
    return -1;
  }

  auto* list = FirmwareInfoListCreate(res.unwrap());
  *out = list;
  return size;
}
void AUTDGetFirmwareInfo(void* const p_firm_info_list, const int32_t index, char* cpu_ver, char* fpga_ver) {
  auto* wrapper = static_cast<FirmwareInfoListWrapper*>(p_firm_info_list);
  const auto& cpu_ver_ = wrapper->list[index].cpu_version();
  const auto& fpga_ver_ = wrapper->list[index].fpga_version();
  std::char_traits<char>::copy(cpu_ver, cpu_ver_.c_str(), cpu_ver_.size() + 1);
  std::char_traits<char>::copy(fpga_ver, fpga_ver_.c_str(), fpga_ver_.size() + 1);
}
void AUTDFreeFirmwareInfoListPointer(void* const p_firm_info_list) {
  auto* wrapper = static_cast<FirmwareInfoListWrapper*>(p_firm_info_list);
  FirmwareInfoListDelete(wrapper);
}
int32_t AUTDGetLastError(char* error) {
  const auto& error_ = LastError();
  const auto size = static_cast<int32_t>(error_.size() + 1);
  if (error == nullptr) return size;
  std::char_traits<char>::copy(error, error_.c_str(), size);
  return size;
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
float AUTDWavelength(void* const handle) {
  auto* cnt = static_cast<ControllerWrapper*>(handle);
  return static_cast<float>(cnt->ptr->geometry()->wavelength());
}
void AUTDSetWavelength(void* const handle, const float wavelength) {
  auto* cnt = static_cast<ControllerWrapper*>(handle);
  cnt->ptr->geometry()->set_wavelength(static_cast<double>(wavelength));
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
  return cnt->ptr->remaining_in_buffer();
}
#pragma endregion

#pragma region Gain
void AUTDNullGain(void** gain) {
  auto* g = GainCreate(autd::gain::NullGain::Create());
  *gain = g;
}
void AUTDGroupedGain(void** gain, int32_t* const group_ids, void** in_gains, const int32_t size) {
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
void AUTDDeleteGain(void* const gain) {
  auto* g = static_cast<GainWrapper*>(gain);
  GainDelete(g);
}
void AUTDFocalPointGain(void** gain, const float x, const float y, const float z, const uint8_t duty) {
  auto* g =
      GainCreate(autd::gain::FocalPointGain::Create(autd::Vector3(static_cast<double>(x), static_cast<double>(y), static_cast<double>(z)), duty));
  *gain = g;
}
void AUTDBesselBeamGain(void** gain, const float x, const float y, const float z, const float n_x, const float n_y, const float n_z,
                        const float theta_z, const uint8_t duty) {
  auto* g = GainCreate(autd::gain::BesselBeamGain::Create(autd::Vector3(static_cast<double>(x), static_cast<double>(y), static_cast<double>(z)),
                                                          autd::Vector3(static_cast<double>(n_x), static_cast<double>(n_y), static_cast<double>(n_z)),
                                                          static_cast<double>(theta_z), duty));
  *gain = g;
}
void AUTDPlaneWaveGain(void** gain, const float n_x, const float n_y, const float n_z, const uint8_t duty) {
  auto* g = GainCreate(
      autd::gain::PlaneWaveGain::Create(autd::Vector3(static_cast<double>(n_x), static_cast<double>(n_y), static_cast<double>(n_z)), duty));
  *gain = g;
}
void AUTDCustomGain(void** gain, uint16_t* data, const int32_t data_length) {
  auto* g = GainCreate(autd::gain::CustomGain::Create(data, data_length));
  *gain = g;
}

void AUTDTransducerTestGain(void** gain, const int32_t idx, const uint8_t duty, const uint8_t phase) {
  auto* g = GainCreate(autd::gain::TransducerTestGain::Create(idx, duty, phase));
  *gain = g;
}
#pragma endregion

#pragma region Modulation
void AUTDModulation(void** mod, const uint8_t amp) {
  auto* m = ModulationCreate(autd::modulation::Modulation::Create(amp));
  *mod = m;
}
void AUTDDeleteModulation(void* const mod) {
  auto* m = static_cast<ModulationWrapper*>(mod);
  ModulationDelete(m);
}
void AUTDCustomModulation(void** mod, uint8_t* buf, const uint32_t size) {
  auto* m = ModulationCreate(autd::modulation::Modulation::Create(0));
  m->ptr->buffer.resize(size, 0);
  std::memcpy(&m->ptr->buffer[0], buf, size);
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
void AUTDSineModulation(void** mod, const int32_t freq, const float amp, const float offset) {
  auto* m = ModulationCreate(autd::modulation::SineModulation::Create(freq, static_cast<double>(amp), static_cast<double>(offset)));
  *mod = m;
}
#pragma endregion

#pragma region LowLevelInterface
bool AUTDAppendGain(void* const handle, void* const gain) {
  auto* cnt = static_cast<ControllerWrapper*>(handle);
  auto* g = static_cast<GainWrapper*>(gain);
  auto res = cnt->ptr->AppendGain(g->ptr);
  if (res.is_err()) LastError() = res.unwrap_err();
  return res.unwrap_or(false);
}

bool AUTDAppendGainSync(void* const handle, void* const gain, const bool wait_for_send) {
  auto* cnt = static_cast<ControllerWrapper*>(handle);
  auto* g = static_cast<GainWrapper*>(gain);
  auto res = cnt->ptr->AppendGainSync(g->ptr, wait_for_send);
  if (res.is_err()) LastError() = res.unwrap_err();
  return res.unwrap_or(false);
}
bool AUTDAppendModulation(void* const handle, void* const mod) {
  auto* cnt = static_cast<ControllerWrapper*>(handle);
  auto* m = static_cast<ModulationWrapper*>(mod);
  auto res = cnt->ptr->AppendModulation(m->ptr);
  if (res.is_err()) LastError() = res.unwrap_err();
  return res.unwrap_or(false);
}
bool AUTDAppendModulationSync(void* const handle, void* const mod) {
  auto* cnt = static_cast<ControllerWrapper*>(handle);
  auto* m = static_cast<ModulationWrapper*>(mod);
  auto res = cnt->ptr->AppendModulationSync(m->ptr);
  if (res.is_err()) LastError() = res.unwrap_err();
  return res.unwrap_or(false);
}
void AUTDAddSTMGain(void* const handle, void* const gain) {
  auto* cnt = static_cast<ControllerWrapper*>(handle);
  auto* g = static_cast<GainWrapper*>(gain);
  cnt->ptr->AddSTMGain(g->ptr);
}
bool AUTDStartSTModulation(void* const handle, const float freq) {
  auto* cnt = static_cast<ControllerWrapper*>(handle);
  auto res = cnt->ptr->StartSTModulation(static_cast<double>(freq));
  if (res.is_err()) LastError() = res.unwrap_err();
  return res.unwrap_or(false);
}
bool AUTDStopSTModulation(void* const handle) {
  auto* cnt = static_cast<ControllerWrapper*>(handle);
  auto res = cnt->ptr->StopSTModulation();
  if (res.is_err()) LastError() = res.unwrap_err();
  return res.unwrap_or(false);
}
bool AUTDFinishSTModulation(void* const handle) {
  auto* cnt = static_cast<ControllerWrapper*>(handle);
  auto res = cnt->ptr->FinishSTModulation();
  if (res.is_err()) LastError() = res.unwrap_err();
  return res.unwrap_or(false);
}
void AUTDFlush(void* const handle) {
  auto* cnt = static_cast<ControllerWrapper*>(handle);
  cnt->ptr->Flush();
}
int32_t AUTDDeviceIdxForTransIdx(void* const handle, const int32_t global_trans_idx) {
  auto* cnt = static_cast<ControllerWrapper*>(handle);
  const auto res = cnt->ptr->geometry()->device_idx_for_trans_idx(global_trans_idx);
  return static_cast<int32_t>(res);
}
void AUTDTransPositionByGlobal(void* const handle, const int32_t global_trans_idx, float* x, float* y, float* z) {
  auto* cnt = static_cast<ControllerWrapper*>(handle);
  const auto pos = cnt->ptr->geometry()->position(global_trans_idx);
  *x = static_cast<float>(pos.x());
  *y = static_cast<float>(pos.y());
  *z = static_cast<float>(pos.z());
}
void AUTDTransPositionByLocal(void* const handle, const int32_t device_idx, const int32_t local_trans_idx, float* x, float* y, float* z) {
  auto* cnt = static_cast<ControllerWrapper*>(handle);
  const auto pos = cnt->ptr->geometry()->position(device_idx, local_trans_idx);
  *x = static_cast<float>(pos.x());
  *y = static_cast<float>(pos.y());
  *z = static_cast<float>(pos.z());
}
void AUTDDeviceDirection(void* const handle, const int32_t device_idx, float* x, float* y, float* z) {
  auto* cnt = static_cast<ControllerWrapper*>(handle);
  const auto dir = cnt->ptr->geometry()->direction(device_idx);
  *x = static_cast<float>(dir.x());
  *y = static_cast<float>(dir.y());
  *z = static_cast<float>(dir.z());
}
#pragma endregion
