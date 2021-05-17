// File: c_api.cpp
// Project: base
// Created Date: 08/03/2021
// Author: Shun Suzuki
// -----
// Last Modified: 17/05/2021
// Modified By: Shun Suzuki (suzuki@hapis.k.u-tokyo.ac.jp)
// -----
// Copyright (c) 2021 Hapis Lab. All rights reserved.
//

#include <cstdint>
#include <cstring>

#include "./autd3_c_api.h"
#include "autd3.hpp"
#include "primitive_gain.hpp"
#include "primitive_modulation.hpp"
#include "wrapper.hpp"
#include "wrapper_gain.hpp"
#include "wrapper_link.hpp"
#include "wrapper_modulation.hpp"

namespace {
std::string& LastError() {
  static std::string msg("");
  return msg;
}

autd::Vector3 ToVec3(const float x, const float y, const float z) {
  return autd::Vector3(static_cast<double>(x), static_cast<double>(y), static_cast<double>(z));
}
autd::Quaternion ToQuaternion(const float w, const float x, const float y, const float z) {
  return autd::Quaternion(static_cast<double>(w), static_cast<double>(x), static_cast<double>(y), static_cast<double>(z));
}

}  // namespace

#pragma region Controller
void AUTDCreateController(void** out) {
  const auto ptr = new autd::Controller();
  auto* cnt = ptr;
  *out = cnt;
}
bool AUTDOpenControllerWith(void* const handle, void* const p_link) {
  auto* cnt = static_cast<autd::Controller*>(handle);
  auto* link = static_cast<LinkWrapper*>(p_link);
  auto res = cnt->OpenWith(link->ptr);
  LinkDelete(link);
  if (res.is_err()) {
    LastError() = res.unwrap_err();
    return false;
  }
  return true;
}
int32_t AUTDAddDevice(void* const handle, const float x, const float y, const float z, const float rz1, const float ry, const float rz2,
                      const int32_t group_id) {
  auto* cnt = static_cast<autd::Controller*>(handle);
  const auto res = cnt->geometry()->AddDevice(ToVec3(x, y, z), ToVec3(rz1, ry, rz2), group_id);
  return static_cast<int32_t>(res);
}
int32_t AUTDAddDeviceQuaternion(void* const handle, const float x, const float y, const float z, const float qua_w, const float qua_x,
                                const float qua_y, const float qua_z, const int32_t group_id) {
  auto* cnt = static_cast<autd::Controller*>(handle);
  const auto res = cnt->geometry()->AddDevice(ToVec3(x, y, z), ToQuaternion(qua_w, qua_x, qua_y, qua_z), group_id);
  return static_cast<int32_t>(res);
}
int32_t AUTDDeleteDevice(void* const handle, const int32_t idx) {
  auto* cnt = static_cast<autd::Controller*>(handle);
  const auto res = cnt->geometry()->DelDevice(static_cast<size_t>(idx));
  return static_cast<int32_t>(res);
}
void AUTDClearDevices(void* const handle) {
  auto* cnt = static_cast<autd::Controller*>(handle);
  cnt->geometry()->ClearDevices();
}
bool AUTDSynchronize(void* const handle, const uint16_t mod_smpl_freq_div, const uint16_t mod_buf_size) {
  auto* cnt = static_cast<autd::Controller*>(handle);
  const autd::core::Configuration config(mod_smpl_freq_div, mod_buf_size);
  if (auto res = cnt->Synchronize(config); res.is_err()) {
    LastError() = res.unwrap_err();
    return false;
  }
  return true;
}
bool AUTDCloseController(void* const handle) {
  auto* cnt = static_cast<autd::Controller*>(handle);
  if (auto res = cnt->Close(); res.is_err()) {
    LastError() = res.unwrap_err();
    return false;
  }
  return true;
}
bool AUTDClear(void* const handle) {
  auto* cnt = static_cast<autd::Controller*>(handle);
  if (auto res = cnt->Clear(); res.is_err()) {
    LastError() = res.unwrap_err();
    return false;
  }
  return true;
}
void AUTDFreeController(void* const handle) {
  auto* cnt = static_cast<autd::Controller*>(handle);
  delete cnt;
}
void AUTDSetSilentMode(void* const handle, const bool mode) {
  auto* cnt = static_cast<autd::Controller*>(handle);
  cnt->silent_mode() = mode;
}
bool AUTDStop(void* const handle) {
  auto* cnt = static_cast<autd::Controller*>(handle);
  if (auto res = cnt->Stop(); res.is_err()) {
    LastError() = res.unwrap_err();
    return false;
  }
  return true;
}
int32_t AUTDGetFirmwareInfoListPointer(void* const handle, void** out) {
  auto* cnt = static_cast<autd::Controller*>(handle);
  const auto size = static_cast<int32_t>(cnt->geometry()->num_devices());
  auto res = cnt->firmware_info_list();
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
  auto* cnt = static_cast<autd::Controller*>(handle);
  return cnt->is_open();
}
bool AUTDIsSilentMode(void* const handle) {
  auto* cnt = static_cast<autd::Controller*>(handle);
  return cnt->silent_mode();
}
float AUTDWavelength(void* const handle) {
  auto* cnt = static_cast<autd::Controller*>(handle);
  return static_cast<float>(cnt->geometry()->wavelength());
}
void AUTDSetWavelength(void* const handle, const float wavelength) {
  auto* cnt = static_cast<autd::Controller*>(handle);
  cnt->geometry()->wavelength() = static_cast<double>(wavelength);
}
int32_t AUTDNumDevices(void* const handle) {
  auto* cnt = static_cast<autd::Controller*>(handle);
  const auto res = cnt->geometry()->num_devices();
  return static_cast<int32_t>(res);
}
int32_t AUTDNumTransducers(void* const handle) {
  auto* cnt = static_cast<autd::Controller*>(handle);
  const auto res = cnt->geometry()->num_transducers();
  return static_cast<int32_t>(res);
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

  auto* g_gain = GainCreate(autd::gain::Grouped::Create(gain_map));

  *gain = g_gain;
}
void AUTDDeleteGain(void* const gain) {
  auto* g = static_cast<GainWrapper*>(gain);
  GainDelete(g);
}
void AUTDFocalPointGain(void** gain, const float x, const float y, const float z, const uint8_t duty) {
  auto* g = GainCreate(autd::gain::FocalPoint::Create(ToVec3(x, y, z), duty));
  *gain = g;
}
void AUTDBesselBeamGain(void** gain, const float x, const float y, const float z, const float n_x, const float n_y, const float n_z,
                        const float theta_z, const uint8_t duty) {
  auto* g = GainCreate(autd::gain::BesselBeam::Create(ToVec3(x, y, z), ToVec3(n_x, n_y, n_z), static_cast<double>(theta_z), duty));
  *gain = g;
}
void AUTDPlaneWaveGain(void** gain, const float n_x, const float n_y, const float n_z, const uint8_t duty) {
  auto* g = GainCreate(autd::gain::PlaneWave::Create(ToVec3(n_x, n_y, n_z), duty));
  *gain = g;
}
void AUTDCustomGain(void** gain, uint16_t* data, const int32_t data_length) {
  auto* g = GainCreate(autd::gain::Custom::Create(data, data_length));
  *gain = g;
}

void AUTDTransducerTestGain(void** gain, const int32_t idx, const uint8_t duty, const uint8_t phase) {
  auto* g = GainCreate(autd::gain::TransducerTest::Create(idx, duty, phase));
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
  m->ptr->buffer().resize(size, 0);
  std::memcpy(&m->ptr->buffer()[0], buf, size);
  *mod = m;
}
void AUTDSquareModulation(void** mod, const int32_t freq, const uint8_t low, const uint8_t high) {
  auto* m = ModulationCreate(autd::modulation::Square::Create(freq, low, high));
  *mod = m;
}
void AUTDSawModulation(void** mod, const int32_t freq) {
  auto* m = ModulationCreate(autd::modulation::Saw::Create(freq));
  *mod = m;
}
void AUTDSineModulation(void** mod, const int32_t freq, const float amp, const float offset) {
  auto* m = ModulationCreate(autd::modulation::Sine::Create(freq, static_cast<double>(amp), static_cast<double>(offset)));
  *mod = m;
}
#pragma endregion

#pragma region LowLevelInterface
bool AUTDSend(void* const handle, void* const gain, void* const modulation) {
  auto* cnt = static_cast<autd::Controller*>(handle);
  const auto g = gain == nullptr ? nullptr : static_cast<GainWrapper*>(gain)->ptr;
  const auto m = modulation == nullptr ? nullptr : static_cast<ModulationWrapper*>(modulation)->ptr;
  if (auto res = cnt->Send(g, m); res.is_err()) {
    LastError() = res.unwrap_err();
    return false;
  }
  return true;
}
void AUTDAddSTMGain(void* const handle, void* const gain) {
  auto* cnt = static_cast<autd::Controller*>(handle);
  auto* g = static_cast<GainWrapper*>(gain);
  cnt->stm()->AddGain(g->ptr);
}
bool AUTDStartSTM(void* const handle, const float freq) {
  auto* cnt = static_cast<autd::Controller*>(handle);
  if (auto res = cnt->stm()->Start(static_cast<double>(freq)); res.is_err()) {
    LastError() = res.unwrap_err();
    return false;
  }
  return true;
}
bool AUTDStopSTM(void* const handle) {
  auto* cnt = static_cast<autd::Controller*>(handle);
  if (auto res = cnt->stm()->Stop(); res.is_err()) {
    LastError() = res.unwrap_err();
    return false;
  }
  return true;
}
bool AUTDFinishSTM(void* const handle) {
  auto* cnt = static_cast<autd::Controller*>(handle);
  if (auto res = cnt->stm()->Finish(); res.is_err()) {
    LastError() = res.unwrap_err();
    return false;
  }
  return true;
}
int32_t AUTDDeviceIdxForTransIdx(void* const handle, const int32_t global_trans_idx) {
  auto* cnt = static_cast<autd::Controller*>(handle);
  const auto res = cnt->geometry()->device_idx_for_trans_idx(global_trans_idx);
  return static_cast<int32_t>(res);
}
void AUTDTransPositionByGlobal(void* const handle, const int32_t global_trans_idx, float* x, float* y, float* z) {
  auto* cnt = static_cast<autd::Controller*>(handle);
  const auto pos = cnt->geometry()->position(global_trans_idx);
  *x = static_cast<float>(pos.x());
  *y = static_cast<float>(pos.y());
  *z = static_cast<float>(pos.z());
}
void AUTDTransPositionByLocal(void* const handle, const int32_t device_idx, const int32_t local_trans_idx, float* x, float* y, float* z) {
  auto* cnt = static_cast<autd::Controller*>(handle);
  const auto pos = cnt->geometry()->position(device_idx, local_trans_idx);
  *x = static_cast<float>(pos.x());
  *y = static_cast<float>(pos.y());
  *z = static_cast<float>(pos.z());
}
void AUTDDeviceDirection(void* const handle, const int32_t device_idx, float* x, float* y, float* z) {
  auto* cnt = static_cast<autd::Controller*>(handle);
  const auto dir = cnt->geometry()->direction(device_idx);
  *x = static_cast<float>(dir.x());
  *y = static_cast<float>(dir.y());
  *z = static_cast<float>(dir.z());
}
#pragma endregion
