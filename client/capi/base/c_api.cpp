﻿// File: c_api.cpp
// Project: base
// Created Date: 08/03/2021
// Author: Shun Suzuki
// -----
// Last Modified: 03/06/2021
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
#include "primitive_sequence.hpp"
#include "wrapper.hpp"
#include "wrapper_link.hpp"

namespace {
std::string& LastError() {
  static std::string msg("");
  return msg;
}
autd::Vector3 ToVec3(const double x, const double y, const double z) { return autd::Vector3(x, y, z); }
autd::Quaternion ToQuaternion(const double w, const double x, const double y, const double z) { return autd::Quaternion(w, x, y, z); }
}  // namespace

void AUTDCreateController(void** out) { *out = ControllerCreate(autd::Controller::create()); }
bool AUTDOpenController(void* const handle, void* const p_link) {
  auto* wrapper = static_cast<ControllerWrapper*>(handle);
  auto* link = static_cast<LinkWrapper*>(p_link);
  auto res = wrapper->ptr->open(std::move(link->ptr));
  LinkDelete(link);
  if (res.is_err()) {
    LastError() = res.unwrap_err();
    return false;
  }
  return true;
}
int32_t AUTDAddDevice(void* const handle, const double x, const double y, const double z, const double rz1, const double ry, const double rz2,
                      const int32_t gid) {
  auto* wrapper = static_cast<ControllerWrapper*>(handle);
  const auto res = wrapper->ptr->geometry()->add_device(ToVec3(x, y, z), ToVec3(rz1, ry, rz2), gid);
  return static_cast<int32_t>(res);
}
int32_t AUTDAddDeviceQuaternion(void* const handle, const double x, const double y, const double z, const double qw, const double qx, const double qy,
                                const double qz, const int32_t gid) {
  auto* wrapper = static_cast<ControllerWrapper*>(handle);
  const auto res = wrapper->ptr->geometry()->add_device(ToVec3(x, y, z), ToQuaternion(qw, qx, qy, qz), gid);
  return static_cast<int32_t>(res);
}
int32_t AUTDDeleteDevice(void* const handle, const int32_t idx) {
  auto* wrapper = static_cast<ControllerWrapper*>(handle);
  const auto res = wrapper->ptr->geometry()->del_device(static_cast<size_t>(idx));
  return static_cast<int32_t>(res);
}
void AUTDClearDevices(void* const handle) {
  auto* wrapper = static_cast<ControllerWrapper*>(handle);
  wrapper->ptr->geometry()->clear_devices();
}
bool AUTDSynchronize(void* const handle, const uint16_t mod_smpl_freq_div, const uint16_t mod_buf_size) {
  auto* wrapper = static_cast<ControllerWrapper*>(handle);
  const autd::core::Configuration config(mod_smpl_freq_div, mod_buf_size);
  if (auto res = wrapper->ptr->synchronize(config); res.is_err()) {
    LastError() = res.unwrap_err();
    return false;
  }
  return true;
}
bool AUTDCloseController(void* const handle) {
  auto* wrapper = static_cast<ControllerWrapper*>(handle);
  if (auto res = wrapper->ptr->close(); res.is_err()) {
    LastError() = res.unwrap_err();
    return false;
  }
  return true;
}
bool AUTDClear(void* const handle) {
  auto* wrapper = static_cast<ControllerWrapper*>(handle);
  if (auto res = wrapper->ptr->clear(); res.is_err()) {
    LastError() = res.unwrap_err();
    return false;
  }
  return true;
}
void AUTDFreeController(void* const handle) {
  auto* wrapper = static_cast<ControllerWrapper*>(handle);
  ControllerDelete(wrapper);
}

bool AUTDIsOpen(void* const handle) {
  auto* wrapper = static_cast<ControllerWrapper*>(handle);
  return wrapper->ptr->is_open();
}
bool AUTDIsSilentMode(void* const handle) {
  auto* wrapper = static_cast<ControllerWrapper*>(handle);
  return wrapper->ptr->silent_mode();
}
bool AUTDIsForceFan(void* const handle) {
  auto* wrapper = static_cast<ControllerWrapper*>(handle);
  return wrapper->ptr->force_fan();
}
void AUTDSetSilentMode(void* const handle, const bool mode) {
  auto* wrapper = static_cast<ControllerWrapper*>(handle);
  wrapper->ptr->silent_mode() = mode;
}
void AUTDSetForceFan(void* const handle, const bool force) {
  auto* wrapper = static_cast<ControllerWrapper*>(handle);
  wrapper->ptr->force_fan() = force;
}
void AUTDSetReadFPGAInfo(void* const handle, const bool reads_fpga_info) {
  auto* wrapper = static_cast<ControllerWrapper*>(handle);
  wrapper->ptr->reads_fpga_info() = reads_fpga_info;
}
double AUTDWavelength(void* const handle) {
  auto* wrapper = static_cast<ControllerWrapper*>(handle);
  return wrapper->ptr->geometry()->wavelength();
}
void AUTDSetWavelength(void* const handle, const double wavelength) {
  auto* wrapper = static_cast<ControllerWrapper*>(handle);
  wrapper->ptr->geometry()->wavelength() = wavelength;
}
bool AUTDReadFPGAInfo(void* handle, uint8_t* out) {
  auto* wrapper = static_cast<ControllerWrapper*>(handle);
  auto res = wrapper->ptr->fpga_info();
  if (res.is_err()) {
    LastError() = res.unwrap_err();
    return false;
  }
  auto fpga_infos = res.unwrap();
  std::memcpy(out, &fpga_infos[0], fpga_infos.size());
  return true;
}
bool AUTDUpdateCtrlFlags(void* handle) {
  auto* wrapper = static_cast<ControllerWrapper*>(handle);
  if (auto res = wrapper->ptr->update_ctrl_flag(); res.is_err()) {
    LastError() = res.unwrap_err();
    return false;
  }
  return true;
}
bool AUTDSetOutputDelay(void* handle, uint16_t* delay) {
  auto* wrapper = static_cast<ControllerWrapper*>(handle);
  const auto num_devices = wrapper->ptr->geometry()->num_devices();
  std::vector<autd::DataArray> delay_;
  delay_.resize(num_devices);
  for (size_t i = 0; i < num_devices; i++)
    std::memcpy(&delay_[i][0], &delay[i * autd::core::NUM_TRANS_IN_UNIT], sizeof(uint16_t) * autd::core::NUM_TRANS_IN_UNIT);
  if (auto res = wrapper->ptr->set_output_delay(delay_); res.is_err()) {
    LastError() = res.unwrap_err();
    return false;
  }
  return true;
}

int32_t AUTDNumDevices(void* const handle) {
  auto* wrapper = static_cast<ControllerWrapper*>(handle);
  const auto res = wrapper->ptr->geometry()->num_devices();
  return static_cast<int32_t>(res);
}
int32_t AUTDNumTransducers(void* const handle) {
  auto* wrapper = static_cast<ControllerWrapper*>(handle);
  const auto res = wrapper->ptr->geometry()->num_transducers();
  return static_cast<int32_t>(res);
}
int32_t AUTDDeviceIdxForTransIdx(void* const handle, const int32_t global_trans_idx) {
  auto* wrapper = static_cast<ControllerWrapper*>(handle);
  const auto res = wrapper->ptr->geometry()->device_idx_for_trans_idx(global_trans_idx);
  return static_cast<int32_t>(res);
}
void AUTDTransPositionByGlobal(void* const handle, const int32_t global_trans_idx, double* x, double* y, double* z) {
  auto* wrapper = static_cast<ControllerWrapper*>(handle);
  const auto pos = wrapper->ptr->geometry()->position(global_trans_idx);
  *x = pos.x();
  *y = pos.y();
  *z = pos.z();
}
void AUTDTransPositionByLocal(void* const handle, const int32_t device_idx, const int32_t local_trans_idx, double* x, double* y, double* z) {
  auto* wrapper = static_cast<ControllerWrapper*>(handle);
  const auto pos = wrapper->ptr->geometry()->position(device_idx, local_trans_idx);
  *x = pos.x();
  *y = pos.y();
  *z = pos.z();
}

void AUTDDeviceXDirection(void* const handle, const int32_t device_idx, double* x, double* y, double* z) {
  auto* wrapper = static_cast<ControllerWrapper*>(handle);
  const auto dir = wrapper->ptr->geometry()->x_direction(device_idx);
  *x = dir.x();
  *y = dir.y();
  *z = dir.z();
}
void AUTDDeviceYDirection(void* const handle, const int32_t device_idx, double* x, double* y, double* z) {
  auto* wrapper = static_cast<ControllerWrapper*>(handle);
  const auto dir = wrapper->ptr->geometry()->y_direction(device_idx);
  *x = dir.x();
  *y = dir.y();
  *z = dir.z();
}
void AUTDDeviceZDirection(void* const handle, const int32_t device_idx, double* x, double* y, double* z) {
  auto* wrapper = static_cast<ControllerWrapper*>(handle);
  const auto dir = wrapper->ptr->geometry()->z_direction(device_idx);
  *x = dir.x();
  *y = dir.y();
  *z = dir.z();
}

int32_t AUTDGetFirmwareInfoListPointer(void* const handle, void** out) {
  auto* wrapper = static_cast<ControllerWrapper*>(handle);
  const auto size = static_cast<int32_t>(wrapper->ptr->geometry()->num_devices());
  auto res = wrapper->ptr->firmware_info_list();
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

void AUTDNullGain(void** gain) {
  auto* g = GainCreate(autd::gain::NullGain::create());
  *gain = g;
}
void AUTDGroupedGain(void** gain) {
  auto* g = GainCreate(autd::gain::Grouped::create());
  *gain = g;
}

void AUTDGroupedGainAdd(void* grouped_gain, const int32_t id, void* gain) {
  auto* gg = static_cast<GainWrapper*>(grouped_gain);
  auto* ing = static_cast<GainWrapper*>(gain);
  auto* pgg = dynamic_cast<autd::gain::Grouped*>(gg->ptr.get());
  pgg->add(id, ing->ptr);
}

void AUTDFocalPointGain(void** gain, const double x, const double y, const double z, const uint8_t duty) {
  auto* g = GainCreate(autd::gain::FocalPoint::create(ToVec3(x, y, z), duty));
  *gain = g;
}
void AUTDBesselBeamGain(void** gain, const double x, const double y, const double z, const double n_x, const double n_y, const double n_z,
                        const double theta_z, const uint8_t duty) {
  auto* g = GainCreate(autd::gain::BesselBeam::create(ToVec3(x, y, z), ToVec3(n_x, n_y, n_z), theta_z, duty));
  *gain = g;
}
void AUTDPlaneWaveGain(void** gain, const double n_x, const double n_y, const double n_z, const uint8_t duty) {
  auto* g = GainCreate(autd::gain::PlaneWave::create(ToVec3(n_x, n_y, n_z), duty));
  *gain = g;
}
void AUTDCustomGain(void** gain, uint16_t* data, const int32_t data_length) {
  auto* g = GainCreate(autd::gain::Custom::create(data, data_length));
  *gain = g;
}
void AUTDTransducerTestGain(void** gain, const int32_t idx, const uint8_t duty, const uint8_t phase) {
  auto* g = GainCreate(autd::gain::TransducerTest::create(idx, duty, phase));
  *gain = g;
}
void AUTDDeleteGain(void* const gain) {
  auto* g = static_cast<GainWrapper*>(gain);
  GainDelete(g);
}

void AUTDStaticModulation(void** mod, const uint8_t amp) {
  auto* m = ModulationCreate(autd::modulation::Modulation::create(amp));
  *mod = m;
}
void AUTDCustomModulation(void** mod, uint8_t* buf, const uint32_t size) {
  auto* m = ModulationCreate(autd::modulation::Modulation::create(0));
  m->ptr->buffer().resize(size, 0);
  std::memcpy(&m->ptr->buffer()[0], buf, size);
  *mod = m;
}
void AUTDSquareModulation(void** mod, const int32_t freq, const uint8_t low, const uint8_t high) {
  auto* m = ModulationCreate(autd::modulation::Square::create(freq, low, high));
  *mod = m;
}
void AUTDSawModulation(void** mod, const int32_t freq) {
  auto* m = ModulationCreate(autd::modulation::Saw::create(freq));
  *mod = m;
}
void AUTDSineModulation(void** mod, const int32_t freq, const double amp, const double offset) {
  auto* m = ModulationCreate(autd::modulation::Sine::create(freq, amp, offset));
  *mod = m;
}
void AUTDDeleteModulation(void* const mod) {
  auto* m = static_cast<ModulationWrapper*>(mod);
  ModulationDelete(m);
}

void AUTDSequence(void** out) {
  auto* s = SequenceCreate(autd::sequence::PointSequence::create());
  *out = s;
}
bool AUTDSequenceAddPoint(void* const seq, const double x, const double y, const double z) {
  auto* seq_w = static_cast<SequenceWrapper*>(seq);
  if (auto res = seq_w->ptr->add_point(ToVec3(x, y, z)); res.is_err()) {
    LastError() = res.unwrap_err();
    return false;
  }
  return true;
}
bool AUTDSequenceAddPoints(void* const seq, double* points, const uint64_t size) {
  auto* seq_w = static_cast<SequenceWrapper*>(seq);
  std::vector<autd::Vector3> p;
  for (size_t i = 0; i < size; i++) p.emplace_back(ToVec3(points[3 * i], points[3 * i + 1], points[3 * i + 2]));
  if (auto res = seq_w->ptr->add_points(p); res.is_err()) {
    LastError() = res.unwrap_err();
    return false;
  }
  return true;
}
double AUTDSequenceSetFreq(void* const seq, const double freq) {
  auto* seq_w = static_cast<SequenceWrapper*>(seq);
  return seq_w->ptr->set_frequency(freq);
}
double AUTDSequenceFreq(void* const seq) {
  auto* seq_w = static_cast<SequenceWrapper*>(seq);
  return seq_w->ptr->frequency();
}
uint32_t AUTDSequencePeriod(void* seq) {
  auto* seq_w = static_cast<SequenceWrapper*>(seq);
  return static_cast<uint32_t>(seq_w->ptr->period_us());
}
uint32_t AUTDSequenceSamplingPeriod(void* seq) {
  auto* seq_w = static_cast<SequenceWrapper*>(seq);
  return static_cast<uint32_t>(seq_w->ptr->sampling_period_us());
}
double AUTDSequenceSamplingFreq(void* const seq) {
  auto* seq_w = static_cast<SequenceWrapper*>(seq);
  return seq_w->ptr->sampling_frequency();
}
uint16_t AUTDSequenceSamplingFreqDiv(void* const seq) {
  auto* seq_w = static_cast<SequenceWrapper*>(seq);
  return seq_w->ptr->sampling_frequency_division();
}
void AUTDDeleteSequence(void* const seq) {
  auto* seq_w = static_cast<SequenceWrapper*>(seq);
  SequenceDelete(seq_w);
}
void AUTDCircumSequence(void** out, const double x, const double y, const double z, const double nx, const double ny, const double nz,
                        const double radius, const uint64_t n) {
  auto* s = SequenceCreate(autd::sequence::Circumference::create(ToVec3(x, y, z), ToVec3(nx, ny, nz), radius, n));
  *out = s;
}

bool AUTDStop(void* const handle) {
  auto* wrapper = static_cast<ControllerWrapper*>(handle);
  if (auto res = wrapper->ptr->stop(); res.is_err()) {
    LastError() = res.unwrap_err();
    return false;
  }
  return true;
}
bool AUTDSendGain(void* const handle, void* const gain) {
  auto* wrapper = static_cast<ControllerWrapper*>(handle);
  const auto g = gain == nullptr ? nullptr : static_cast<GainWrapper*>(gain)->ptr;
  if (auto res = wrapper->ptr->send(g); res.is_err()) {
    LastError() = res.unwrap_err();
    return false;
  }
  return true;
}
bool AUTDSendModulation(void* const handle, void* const mod) {
  auto* wrapper = static_cast<ControllerWrapper*>(handle);
  const auto m = mod == nullptr ? nullptr : static_cast<ModulationWrapper*>(mod)->ptr;
  if (auto res = wrapper->ptr->send(m); res.is_err()) {
    LastError() = res.unwrap_err();
    return false;
  }
  return true;
}
bool AUTDSendGainModulation(void* const handle, void* const gain, void* const mod) {
  auto* wrapper = static_cast<ControllerWrapper*>(handle);
  const auto g = gain == nullptr ? nullptr : static_cast<GainWrapper*>(gain)->ptr;
  const auto m = mod == nullptr ? nullptr : static_cast<ModulationWrapper*>(mod)->ptr;
  if (auto res = wrapper->ptr->send(g, m); res.is_err()) {
    LastError() = res.unwrap_err();
    return false;
  }
  return true;
}
bool AUTDSendSequence(void* const handle, void* const seq) {
  auto* wrapper = static_cast<ControllerWrapper*>(handle);
  const auto s = seq == nullptr ? nullptr : static_cast<SequenceWrapper*>(seq)->ptr;
  if (auto res = wrapper->ptr->send(s); res.is_err()) {
    LastError() = res.unwrap_err();
    return false;
  }
  return true;
}

void AUTDSTMController(void** out, void* handle) {
  auto* wrapper = static_cast<ControllerWrapper*>(handle);
  *out = STMControllerCreate(wrapper->ptr->stm());
}

bool AUTDAddSTMGain(void* const handle, void* const gain) {
  auto* wrapper = static_cast<STMControllerWrapper*>(handle);
  auto* g = static_cast<GainWrapper*>(gain);
  if (auto res = wrapper->ptr->add_gain(g->ptr); res.is_err()) {
    LastError() = res.unwrap_err();
    return false;
  }
  return true;
}

bool AUTDStartSTM(void* const handle, const double freq) {
  auto* wrapper = static_cast<STMControllerWrapper*>(handle);
  if (auto res = wrapper->ptr->start(freq); res.is_err()) {
    LastError() = res.unwrap_err();
    return false;
  }
  return true;
}
bool AUTDStopSTM(void* const handle) {
  auto* wrapper = static_cast<STMControllerWrapper*>(handle);
  if (auto res = wrapper->ptr->stop(); res.is_err()) {
    LastError() = res.unwrap_err();
    return false;
  }
  return true;
}
bool AUTDFinishSTM(void* const handle) {
  auto* wrapper = static_cast<STMControllerWrapper*>(handle);
  if (auto res = wrapper->ptr->finish(); res.is_err()) {
    LastError() = res.unwrap_err();
    return false;
  }
  return true;
}
