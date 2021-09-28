// File: c_api.cpp
// Project: base
// Created Date: 08/03/2021
// Author: Shun Suzuki
// -----
// Last Modified: 28/09/2021
// Modified By: Shun Suzuki (suzuki@hapis.k.u-tokyo.ac.jp)
// -----
// Copyright (c) 2021 Hapis Lab. All rights reserved.
//

#include <cstdint>
#include <cstring>

#include "./autd3_c_api.h"
#include "autd3.hpp"
#include "autd3/gain/primitive.hpp"
#include "autd3/modulation/primitive.hpp"
#include "autd3/sequence/primitive.hpp"
#include "wrapper.hpp"
#include "wrapper_link.hpp"
#include "wrapper_modulation.hpp"

#define AUTD3_CAPI_TRY(action)    \
  try {                           \
    action;                       \
  } catch (std::exception & ex) { \
    LastError() = ex.what();      \
    return false;                 \
  }

#define AUTD3_CAPI_TRY2(action)   \
  try {                           \
    action;                       \
  } catch (std::exception & ex) { \
    LastError() = ex.what();      \
    return -1;                    \
  }

namespace {
std::string& LastError() {
  static std::string msg("");
  return msg;
}
autd::Vector3 ToVec3(const double x, const double y, const double z) { return autd::Vector3(x, y, z); }
autd::Quaternion ToQuaternion(const double w, const double x, const double y, const double z) { return autd::Quaternion(w, x, y, z); }
}  // namespace

void AUTDCreateController(void** out) { *out = ControllerCreate(autd::Controller::create()); }
bool AUTDOpenController(const void* const handle, void* const p_link) {
  const auto* wrapper = static_cast<const ControllerWrapper*>(handle);
  auto* link = static_cast<LinkWrapper*>(p_link);
  auto link_ = std::move(link->ptr);
  LinkDelete(link);
  AUTD3_CAPI_TRY({
    wrapper->ptr->open(std::move(link_));
    return true;
  })
}

int32_t AUTDAddDevice(const void* const handle, const double x, const double y, const double z, const double rz1, const double ry, const double rz2,
                      const int32_t gid) {
  const auto* wrapper = static_cast<const ControllerWrapper*>(handle);
  const auto res = wrapper->ptr->geometry()->add_device(ToVec3(x, y, z), ToVec3(rz1, ry, rz2), gid);
  return static_cast<int32_t>(res);
}
int32_t AUTDAddDeviceQuaternion(const void* const handle, const double x, const double y, const double z, const double qw, const double qx,
                                const double qy, const double qz, const int32_t gid) {
  const auto* wrapper = static_cast<const ControllerWrapper*>(handle);
  const auto res = wrapper->ptr->geometry()->add_device(ToVec3(x, y, z), ToQuaternion(qw, qx, qy, qz), gid);
  return static_cast<int32_t>(res);
}
int32_t AUTDDeleteDevice(const void* const handle, const int32_t idx) {
  const auto* wrapper = static_cast<const ControllerWrapper*>(handle);
  const auto res = wrapper->ptr->geometry()->del_device(static_cast<size_t>(idx));
  return static_cast<int32_t>(res);
}
void AUTDClearDevices(const void* const handle) {
  const auto* wrapper = static_cast<const ControllerWrapper*>(handle);
  wrapper->ptr->geometry()->clear_devices();
}
int32_t AUTDCloseController(const void* const handle) {
  const auto* wrapper = static_cast<const ControllerWrapper*>(handle);
  AUTD3_CAPI_TRY2(return wrapper->ptr->close() ? 1 : 0)
}
int32_t AUTDClear(const void* const handle) {
  const auto* wrapper = static_cast<const ControllerWrapper*>(handle);
  AUTD3_CAPI_TRY2(return wrapper->ptr->clear() ? 1 : 0)
}
void AUTDFreeController(const void* const handle) {
  const auto* wrapper = static_cast<const ControllerWrapper*>(handle);
  ControllerDelete(wrapper);
}
bool AUTDIsOpen(const void* const handle) {
  const auto* wrapper = static_cast<const ControllerWrapper*>(handle);
  return wrapper->ptr->is_open();
}
bool AUTDIsSilentMode(const void* const handle) {
  const auto* wrapper = static_cast<const ControllerWrapper*>(handle);
  return wrapper->ptr->silent_mode();
}
bool AUTDIsForceFan(const void* const handle) {
  const auto* wrapper = static_cast<const ControllerWrapper*>(handle);
  return wrapper->ptr->force_fan();
}
bool AUTDIsReadsFPGAInfo(const void* const handle) {
  const auto* wrapper = static_cast<const ControllerWrapper*>(handle);
  return wrapper->ptr->reads_fpga_info();
}
bool AUTDIsOutputBalance(const void* const handle) {
  const auto* wrapper = static_cast<const ControllerWrapper*>(handle);
  return wrapper->ptr->output_balance();
}
bool AUTDIsHandShake(const void* const handle) {
  const auto* wrapper = static_cast<const ControllerWrapper*>(handle);
  return wrapper->ptr->hand_shake();
}
void AUTDSetSilentMode(const void* const handle, const bool mode) {
  const auto* wrapper = static_cast<const ControllerWrapper*>(handle);
  wrapper->ptr->silent_mode() = mode;
}
void AUTDSetForceFan(const void* const handle, const bool force) {
  const auto* wrapper = static_cast<const ControllerWrapper*>(handle);
  wrapper->ptr->force_fan() = force;
}
void AUTDSetReadsFPGAInfo(const void* const handle, const bool reads_fpga_info) {
  const auto* wrapper = static_cast<const ControllerWrapper*>(handle);
  wrapper->ptr->reads_fpga_info() = reads_fpga_info;
}
void AUTDSetOutputBalance(const void* const handle, const bool output_balance) {
  const auto* wrapper = static_cast<const ControllerWrapper*>(handle);
  wrapper->ptr->output_balance() = output_balance;
}
void AUTDSetHandShake(const void* const handle, const bool hand_shake) {
  const auto* wrapper = static_cast<const ControllerWrapper*>(handle);
  wrapper->ptr->hand_shake() = hand_shake;
}
double AUTDGetWavelength(const void* const handle) {
  const auto* wrapper = static_cast<const ControllerWrapper*>(handle);
  return wrapper->ptr->geometry()->wavelength();
}
double AUTDGetAttenuation(const void* const handle) {
  const auto* wrapper = static_cast<const ControllerWrapper*>(handle);
  return wrapper->ptr->geometry()->attenuation_coefficient();
}
void AUTDSetWavelength(const void* const handle, const double wavelength) {
  const auto* wrapper = static_cast<const ControllerWrapper*>(handle);
  wrapper->ptr->geometry()->wavelength() = wavelength;
}
void AUTDSetAttenuation(const void* const handle, const double attenuation) {
  const auto* wrapper = static_cast<const ControllerWrapper*>(handle);
  wrapper->ptr->geometry()->attenuation_coefficient() = attenuation;
}
bool AUTDGetFPGAInfo(const void* handle, uint8_t* out) {
  const auto* wrapper = static_cast<const ControllerWrapper*>(handle);
  AUTD3_CAPI_TRY({
    const auto res = wrapper->ptr->fpga_info();
    std::memcpy(out, &res[0], res.size());
    return true;
  })
}
int32_t AUTDUpdateCtrlFlags(const void* handle) {
  const auto* wrapper = static_cast<const ControllerWrapper*>(handle);
  AUTD3_CAPI_TRY2(return wrapper->ptr->update_ctrl_flag() ? 1 : 0)
}

int32_t AUTDSetOutputDelay(const void* handle, const uint8_t* const delay) {
  const auto* wrapper = static_cast<const ControllerWrapper*>(handle);
  const auto num_devices = wrapper->ptr->geometry()->num_devices();
  std::vector<std::array<uint8_t, autd::core::NUM_TRANS_IN_UNIT>> delay_;
  delay_.resize(num_devices);
  for (size_t i = 0; i < num_devices; i++) std::memcpy(&delay_[i][0], &delay[i * autd::core::NUM_TRANS_IN_UNIT], autd::core::NUM_TRANS_IN_UNIT);
  AUTD3_CAPI_TRY2(return wrapper->ptr->set_output_delay(delay_) ? 1 : 0)
}
int32_t AUTDSetDutyOffset(const void* handle, const uint8_t* const offset) {
  const auto* wrapper = static_cast<const ControllerWrapper*>(handle);
  const auto num_devices = wrapper->ptr->geometry()->num_devices();
  std::vector<std::array<uint8_t, autd::core::NUM_TRANS_IN_UNIT>> offset_;
  offset_.resize(num_devices);
  for (size_t i = 0; i < num_devices; i++) std::memcpy(&offset_[i][0], &offset[i * autd::core::NUM_TRANS_IN_UNIT], autd::core::NUM_TRANS_IN_UNIT);
  AUTD3_CAPI_TRY2(return wrapper->ptr->set_duty_offset(offset_) ? 1 : 0)
}
int32_t AUTDSetDelayOffset(const void* handle, const uint8_t* const delay, const uint8_t* const offset) {
  const auto* wrapper = static_cast<const ControllerWrapper*>(handle);
  const auto num_devices = wrapper->ptr->geometry()->num_devices();
  std::vector<std::array<uint8_t, autd::core::NUM_TRANS_IN_UNIT>> delay_;
  std::vector<std::array<uint8_t, autd::core::NUM_TRANS_IN_UNIT>> offset_;
  delay_.resize(num_devices);
  offset_.resize(num_devices);
  for (size_t i = 0; i < num_devices; i++) {
    std::memcpy(&delay_[i][0], &delay[i * autd::core::NUM_TRANS_IN_UNIT], autd::core::NUM_TRANS_IN_UNIT);
    std::memcpy(&offset_[i][0], &offset[i * autd::core::NUM_TRANS_IN_UNIT], autd::core::NUM_TRANS_IN_UNIT);
  }
  AUTD3_CAPI_TRY2(return wrapper->ptr->set_delay_offset(delay_, offset_) ? 1 : 0)
}

int32_t AUTDNumDevices(const void* const handle) {
  const auto* wrapper = static_cast<const ControllerWrapper*>(handle);
  const auto res = wrapper->ptr->geometry()->num_devices();
  return static_cast<int32_t>(res);
}
int32_t AUTDNumTransducers(const void* const handle) {
  const auto* wrapper = static_cast<const ControllerWrapper*>(handle);
  const auto res = wrapper->ptr->geometry()->num_transducers();
  return static_cast<int32_t>(res);
}
int32_t AUTDDeviceIdxForTransIdx(const void* const handle, const int32_t global_trans_idx) {
  const auto* wrapper = static_cast<const ControllerWrapper*>(handle);
  const auto res = wrapper->ptr->geometry()->device_idx_for_trans_idx(global_trans_idx);
  return static_cast<int32_t>(res);
}
void AUTDTransPositionByGlobal(const void* const handle, const int32_t global_trans_idx, double* x, double* y, double* z) {
  const auto* wrapper = static_cast<const ControllerWrapper*>(handle);
  const auto& pos = wrapper->ptr->geometry()->position(global_trans_idx);
  *x = pos.x();
  *y = pos.y();
  *z = pos.z();
}
void AUTDTransPositionByLocal(const void* const handle, const int32_t device_idx, const int32_t local_trans_idx, double* x, double* y, double* z) {
  const auto* wrapper = static_cast<const ControllerWrapper*>(handle);
  const auto& pos = wrapper->ptr->geometry()->position(device_idx, local_trans_idx);
  *x = pos.x();
  *y = pos.y();
  *z = pos.z();
}

void AUTDDeviceXDirection(const void* const handle, const int32_t device_idx, double* x, double* y, double* z) {
  const auto* wrapper = static_cast<const ControllerWrapper*>(handle);
  const auto& dir = wrapper->ptr->geometry()->x_direction(device_idx);
  *x = dir.x();
  *y = dir.y();
  *z = dir.z();
}
void AUTDDeviceYDirection(const void* const handle, const int32_t device_idx, double* x, double* y, double* z) {
  const auto* wrapper = static_cast<const ControllerWrapper*>(handle);
  const auto& dir = wrapper->ptr->geometry()->y_direction(device_idx);
  *x = dir.x();
  *y = dir.y();
  *z = dir.z();
}
void AUTDDeviceZDirection(const void* const handle, const int32_t device_idx, double* x, double* y, double* z) {
  const auto* wrapper = static_cast<const ControllerWrapper*>(handle);
  const auto& dir = wrapper->ptr->geometry()->z_direction(device_idx);
  *x = dir.x();
  *y = dir.y();
  *z = dir.z();
}

int32_t AUTDGetFirmwareInfoListPointer(const void* const handle, void** out) {
  const auto* wrapper = static_cast<const ControllerWrapper*>(handle);
  const auto size = static_cast<int32_t>(wrapper->ptr->geometry()->num_devices());
  AUTD3_CAPI_TRY2({
    const auto res = wrapper->ptr->firmware_info_list();
    if (res.empty()) {
      LastError() = "filed to get some infos";
      return -1;
    }
    auto* list = FirmwareInfoListCreate(res);
    *out = list;
    return size;
  })
}
void AUTDGetFirmwareInfo(const void* const p_firm_info_list, const int32_t index, char* cpu_ver, char* fpga_ver) {
  const auto* wrapper = static_cast<const FirmwareInfoListWrapper*>(p_firm_info_list);
  const auto& cpu_ver_ = wrapper->list[index].cpu_version();
  const auto& fpga_ver_ = wrapper->list[index].fpga_version();
  std::char_traits<char>::copy(cpu_ver, cpu_ver_.c_str(), cpu_ver_.size() + 1);
  std::char_traits<char>::copy(fpga_ver, fpga_ver_.c_str(), fpga_ver_.size() + 1);
}
void AUTDFreeFirmwareInfoListPointer(const void* const p_firm_info_list) {
  const auto* wrapper = static_cast<const FirmwareInfoListWrapper*>(p_firm_info_list);
  FirmwareInfoListDelete(wrapper);
}

int32_t AUTDGetLastError(char* error) {
  const auto& error_ = LastError();
  const auto size = static_cast<int32_t>(error_.size() + 1);
  if (error == nullptr) return size;
  std::char_traits<char>::copy(error, error_.c_str(), size);
  return size;
}

void AUTDGainNull(void** gain) {
  auto* g = GainCreate(autd::gain::NullGain::create());
  *gain = g;
}
void AUTDGainGrouped(void** gain) {
  auto* g = GainCreate(autd::gain::Grouped::create());
  *gain = g;
}

void AUTDGainGroupedAdd(const void* grouped_gain, const int32_t id, const void* gain) {
  const auto* gg = static_cast<const GainWrapper*>(grouped_gain);
  const auto* ing = static_cast<const GainWrapper*>(gain);
  auto* pgg = dynamic_cast<autd::gain::Grouped*>(gg->ptr.get());
  pgg->add(id, ing->ptr);
}

void AUTDGainFocalPoint(void** gain, const double x, const double y, const double z, const uint8_t duty) {
  auto* g = GainCreate(autd::gain::FocalPoint::create(ToVec3(x, y, z), duty));
  *gain = g;
}
void AUTDGainBesselBeam(void** gain, const double x, const double y, const double z, const double n_x, const double n_y, const double n_z,
                        const double theta_z, const uint8_t duty) {
  auto* g = GainCreate(autd::gain::BesselBeam::create(ToVec3(x, y, z), ToVec3(n_x, n_y, n_z), theta_z, duty));
  *gain = g;
}
void AUTDGainPlaneWave(void** gain, const double n_x, const double n_y, const double n_z, const uint8_t duty) {
  auto* g = GainCreate(autd::gain::PlaneWave::create(ToVec3(n_x, n_y, n_z), duty));
  *gain = g;
}
void AUTDGainCustom(void** gain, const uint16_t* const data, const int32_t data_length) {
  auto* g = GainCreate(autd::gain::Custom::create(data, data_length));
  *gain = g;
}
void AUTDGainTransducerTest(void** gain, const int32_t idx, const uint8_t duty, const uint8_t phase) {
  auto* g = GainCreate(autd::gain::TransducerTest::create(idx, duty, phase));
  *gain = g;
}
void AUTDDeleteGain(const void* const gain) {
  const auto* g = static_cast<const GainWrapper*>(gain);
  GainDelete(g);
}

void AUTDModulationStatic(void** mod, const uint8_t duty) {
  auto* m = ModulationCreate(autd::modulation::Modulation::create(duty));
  *mod = m;
}
void AUTDModulationCustom(void** mod, const uint8_t* const buf, const uint32_t size) {
  std::vector<uint8_t> buffer;
  for (uint32_t i = 0; i < size; i++) buffer.emplace_back(buf[i]);
  auto* m = ModulationCreate(autd::modulation::Custom::create(buffer));
  *mod = m;
}
void AUTDModulationSquare(void** mod, const int32_t freq, const uint8_t low, const uint8_t high, const double duty) {
  auto* m = ModulationCreate(autd::modulation::Square::create(freq, low, high, duty));
  *mod = m;
}
void AUTDModulationSine(void** mod, const int32_t freq, const double amp, const double offset) {
  auto* m = ModulationCreate(autd::modulation::Sine::create(freq, amp, offset));
  *mod = m;
}
void AUTDModulationSinePressure(void** mod, const int32_t freq, const double amp, const double offset) {
  auto* m = ModulationCreate(autd::modulation::SinePressure::create(freq, amp, offset));
  *mod = m;
}
void AUTDModulationSineLegacy(void** mod, const double freq, const double amp, const double offset) {
  auto* m = ModulationCreate(autd::modulation::SineLegacy::create(freq, amp, offset));
  *mod = m;
}

void AUTDDeleteModulation(const void* const mod) {
  const auto* m = static_cast<const ModulationWrapper*>(mod);
  ModulationDelete(m);
}

void AUTDSequence(void** out) {
  auto* s = SequenceCreate(autd::sequence::PointSequence::create());
  *out = s;
}
void AUTDGainSequence(void** out, const uint16_t gain_mode) {
  auto* s = SequenceCreate(autd::sequence::GainSequence::create(static_cast<autd::GAIN_MODE>(gain_mode)));
  *out = s;
}
bool AUTDSequenceAddPoint(const void* const seq, const double x, const double y, const double z, const uint8_t duty) {
  const auto* seq_w = static_cast<const SequenceWrapper*>(seq);
  AUTD3_CAPI_TRY({
    std::dynamic_pointer_cast<autd::core::PointSequence>(seq_w->ptr)->add_point(ToVec3(x, y, z), duty);
    return true;
  })
}
bool AUTDSequenceAddPoints(const void* const seq, const double* const points, const uint64_t points_size, const uint8_t* const duties,
                           const uint64_t duties_size) {
  const auto* seq_w = static_cast<const SequenceWrapper*>(seq);
  std::vector<autd::Vector3> p;
  for (size_t i = 0; i < points_size; i++) p.emplace_back(ToVec3(points[3 * i], points[3 * i + 1], points[3 * i + 2]));

  std::vector<uint8_t> d;
  for (size_t i = 0; i < duties_size; i++) d.emplace_back(duties[i]);
  AUTD3_CAPI_TRY({
    std::dynamic_pointer_cast<autd::core::PointSequence>(seq_w->ptr)->add_points(p, d);
    return true;
  })
}
bool AUTDSequenceAddGain(const void* const seq, const void* const gain) {
  const auto* seq_w = static_cast<const SequenceWrapper*>(seq);
  const auto g = gain == nullptr ? nullptr : static_cast<const GainWrapper*>(gain)->ptr;
  AUTD3_CAPI_TRY({
    std::dynamic_pointer_cast<autd::core::GainSequence>(seq_w->ptr)->add_gain(g);
    return true;
  })
}
double AUTDSequenceSetFreq(const void* const seq, const double freq) {
  const auto* seq_w = static_cast<const SequenceWrapper*>(seq);
  return seq_w->ptr->set_frequency(freq);
}
double AUTDSequenceFreq(const void* const seq) {
  const auto* seq_w = static_cast<const SequenceWrapper*>(seq);
  return seq_w->ptr->frequency();
}
uint32_t AUTDSequencePeriod(const void* seq) {
  const auto* seq_w = static_cast<const SequenceWrapper*>(seq);
  return static_cast<uint32_t>(seq_w->ptr->period_us());
}
uint32_t AUTDSequenceSamplingPeriod(const void* seq) {
  const auto* seq_w = static_cast<const SequenceWrapper*>(seq);
  return static_cast<uint32_t>(seq_w->ptr->sampling_period_us());
}
double AUTDSequenceSamplingFreq(const void* const seq) {
  const auto* seq_w = static_cast<const SequenceWrapper*>(seq);
  return seq_w->ptr->sampling_frequency();
}
uint16_t AUTDSequenceSamplingFreqDiv(const void* const seq) {
  const auto* seq_w = static_cast<const SequenceWrapper*>(seq);
  return seq_w->ptr->sampling_frequency_division();
}
void AUTDDeleteSequence(const void* const seq) {
  const auto* seq_w = static_cast<const SequenceWrapper*>(seq);
  SequenceDelete(seq_w);
}
void AUTDCircumSequence(void** out, const double x, const double y, const double z, const double nx, const double ny, const double nz,
                        const double radius, const uint64_t n) {
  auto* s = SequenceCreate(autd::sequence::Circumference::create(ToVec3(x, y, z), ToVec3(nx, ny, nz), radius, n));
  *out = s;
}

int32_t AUTDStop(const void* const handle) {
  const auto* wrapper = static_cast<const ControllerWrapper*>(handle);
  AUTD3_CAPI_TRY2(return wrapper->ptr->stop() ? 1 : 0)
}
int32_t AUTDPause(const void* const handle) {
  const auto* wrapper = static_cast<const ControllerWrapper*>(handle);
  AUTD3_CAPI_TRY2(return wrapper->ptr->pause() ? 1 : 0)
}
int32_t AUTDResume(const void* const handle) {
  const auto* wrapper = static_cast<const ControllerWrapper*>(handle);
  AUTD3_CAPI_TRY2(return wrapper->ptr->resume() ? 1 : 0)
}
int32_t AUTDSendGain(const void* const handle, const void* const gain) {
  const auto* wrapper = static_cast<const ControllerWrapper*>(handle);
  const auto g = gain == nullptr ? nullptr : static_cast<const GainWrapper*>(gain)->ptr;
  AUTD3_CAPI_TRY(return wrapper->ptr->send(g) ? 1 : 0)
}
int32_t AUTDSendModulation(const void* const handle, const void* const mod) {
  const auto* wrapper = static_cast<const ControllerWrapper*>(handle);
  const auto m = mod == nullptr ? nullptr : static_cast<const ModulationWrapper*>(mod)->ptr;
  AUTD3_CAPI_TRY(return wrapper->ptr->send(m) ? 1 : 0)
}
int32_t AUTDSendGainModulation(const void* const handle, const void* const gain, const void* const mod) {
  const auto* wrapper = static_cast<const ControllerWrapper*>(handle);
  const auto g = gain == nullptr ? nullptr : static_cast<const GainWrapper*>(gain)->ptr;
  const auto m = mod == nullptr ? nullptr : static_cast<const ModulationWrapper*>(mod)->ptr;
  AUTD3_CAPI_TRY(return wrapper->ptr->send(g, m) ? 1 : 0)
}
int32_t AUTDSendSequence(const void* const handle, const void* const seq) {
  const auto* wrapper = static_cast<const ControllerWrapper*>(handle);
  const auto s = seq == nullptr ? nullptr : static_cast<const SequenceWrapper*>(seq)->ptr;
  AUTD3_CAPI_TRY(return wrapper->ptr->send(std::dynamic_pointer_cast<autd::core::PointSequence>(s)) ? 1 : 0)
}
int32_t AUTDSendGainSequence(const void* const handle, const void* const seq) {
  const auto* wrapper = static_cast<const ControllerWrapper*>(handle);
  const auto s = seq == nullptr ? nullptr : static_cast<const SequenceWrapper*>(seq)->ptr;
  AUTD3_CAPI_TRY(return wrapper->ptr->send(std::dynamic_pointer_cast<autd::core::GainSequence>(s)) ? 1 : 0)
}

void AUTDSTMController(void** out, const void* handle) {
  const auto* wrapper = static_cast<const ControllerWrapper*>(handle);
  *out = STMControllerCreate(wrapper->ptr->stm());
}

bool AUTDAddSTMGain(const void* const handle, const void* const gain) {
  const auto* wrapper = static_cast<const STMControllerWrapper*>(handle);
  const auto* g = static_cast<const GainWrapper*>(gain);
  AUTD3_CAPI_TRY({
    wrapper->ptr->add_gain(g->ptr);
    return true;
  })
}

bool AUTDStartSTM(const void* const handle, const double freq) {
  const auto* wrapper = static_cast<const STMControllerWrapper*>(handle);
  AUTD3_CAPI_TRY({
    wrapper->ptr->start(freq);
    return true;
  })
}
bool AUTDStopSTM(const void* const handle) {
  const auto* wrapper = static_cast<const STMControllerWrapper*>(handle);
  AUTD3_CAPI_TRY({
    wrapper->ptr->stop();
    return true;
  })
}
bool AUTDFinishSTM(const void* const handle) {
  const auto* wrapper = static_cast<const STMControllerWrapper*>(handle);
  AUTD3_CAPI_TRY({
    wrapper->ptr->finish();
    STMControllerDelete(wrapper);
    return true;
  })
}
