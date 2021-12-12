// File: c_api.cpp
// Project: base
// Created Date: 08/03/2021
// Author: Shun Suzuki
// -----
// Last Modified: 12/12/2021
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
#include "custom.hpp"
#include "wrapper.hpp"
#include "wrapper_link.hpp"

#define AUTD3_CAPI_TRY(action)    \
  try {                           \
    action;                       \
  } catch (std::exception & ex) { \
    last_error() = ex.what();     \
    return false;                 \
  }

#define AUTD3_CAPI_TRY2(action)   \
  try {                           \
    action;                       \
  } catch (std::exception & ex) { \
    last_error() = ex.what();     \
    return -1;                    \
  }

namespace {
std::string& last_error() {
  static std::string msg("");  // NOLINT
  return msg;
}
autd::Vector3 to_vec3(const double x, const double y, const double z) { return {x, y, z}; }
autd::Quaternion to_quaternion(const double w, const double x, const double y, const double z) { return {w, x, y, z}; }
}  // namespace

void AUTDCreateController(void** out) { *out = new autd::Controller; }
bool AUTDOpenController(void* const handle, void* const p_link) {
  auto* const wrapper = static_cast<autd::Controller*>(handle);
  auto* link = static_cast<LinkWrapper*>(p_link);
  autd::core::LinkPtr link_ = std::move(link->ptr);
  link_delete(link);
  AUTD3_CAPI_TRY({
    wrapper->open(std::move(link_));
    return true;
  })
}

int32_t AUTDAddDevice(void* const handle, const double x, const double y, const double z, const double rz1, const double ry, const double rz2) {
  auto* const wrapper = static_cast<autd::Controller*>(handle);
  const auto res = wrapper->geometry().add_device(to_vec3(x, y, z), to_vec3(rz1, ry, rz2));
  return static_cast<int32_t>(res);
}
int32_t AUTDAddDeviceQuaternion(void* const handle, const double x, const double y, const double z, const double qw, const double qx, const double qy,
                                const double qz) {
  auto* const wrapper = static_cast<autd::Controller*>(handle);
  const auto res = wrapper->geometry().add_device(to_vec3(x, y, z), to_quaternion(qw, qx, qy, qz));
  return static_cast<int32_t>(res);
}
int32_t AUTDCloseController(void* const handle) {
  auto* const wrapper = static_cast<autd::Controller*>(handle);
  AUTD3_CAPI_TRY2(return wrapper->close() ? 1 : 0)
}
int32_t AUTDClear(void* const handle) {
  auto* wrapper = static_cast<autd::Controller*>(handle);
  AUTD3_CAPI_TRY2(return wrapper->clear() ? 1 : 0)
}
void AUTDFreeController(const void* const handle) {
  const auto* wrapper = static_cast<const autd::Controller*>(handle);
  delete wrapper;
}
bool AUTDIsOpen(const void* const handle) {
  const auto* wrapper = static_cast<const autd::Controller*>(handle);
  return wrapper->is_open();
}
bool AUTDGetOutputEnable(const void* handle) {
  const auto* wrapper = static_cast<const autd::Controller*>(handle);
  return wrapper->output_enable();
}
bool AUTDGetSilentMode(const void* const handle) {
  const auto* wrapper = static_cast<const autd::Controller*>(handle);
  return wrapper->silent_mode();
}
bool AUTDGetForceFan(const void* const handle) {
  const auto* wrapper = static_cast<const autd::Controller*>(handle);
  return wrapper->force_fan();
}
bool AUTDGetReadsFPGAInfo(const void* const handle) {
  const auto* wrapper = static_cast<const autd::Controller*>(handle);
  return wrapper->reads_fpga_info();
}
bool AUTDGetOutputBalance(const void* const handle) {
  const auto* wrapper = static_cast<const autd::Controller*>(handle);
  return wrapper->output_balance();
}
bool AUTDGetCheckAck(const void* const handle) {
  const auto* wrapper = static_cast<const autd::Controller*>(handle);
  return wrapper->check_ack();
}
void AUTDSetOutputEnable(void* const handle, const bool enable) {
  auto* const wrapper = static_cast<autd::Controller*>(handle);
  wrapper->output_enable() = enable;
}
void AUTDSetSilentMode(void* const handle, const bool mode) {
  auto* const wrapper = static_cast<autd::Controller*>(handle);
  wrapper->silent_mode() = mode;
}
void AUTDSetForceFan(void* const handle, const bool force) {
  auto* const wrapper = static_cast<autd::Controller*>(handle);
  wrapper->force_fan() = force;
}
void AUTDSetReadsFPGAInfo(void* const handle, const bool reads_fpga_info) {
  auto* const wrapper = static_cast<autd::Controller*>(handle);
  wrapper->reads_fpga_info() = reads_fpga_info;
}
void AUTDSetOutputBalance(void* const handle, const bool output_balance) {
  auto* const wrapper = static_cast<autd::Controller*>(handle);
  wrapper->output_balance() = output_balance;
}
void AUTDSetCheckAck(void* const handle, const bool check_ack) {
  auto* const wrapper = static_cast<autd::Controller*>(handle);
  wrapper->check_ack() = check_ack;
}
double AUTDGetWavelength(const void* const handle) {
  const auto* wrapper = static_cast<const autd::Controller*>(handle);
  return wrapper->geometry().wavelength();
}
double AUTDGetAttenuation(const void* const handle) {
  const auto* wrapper = static_cast<const autd::Controller*>(handle);
  return wrapper->geometry().attenuation_coefficient();
}
void AUTDSetWavelength(void* const handle, const double wavelength) {
  auto* const wrapper = static_cast<autd::Controller*>(handle);
  wrapper->geometry().wavelength() = wavelength;
}
void AUTDSetAttenuation(void* const handle, const double attenuation) {
  auto* const wrapper = static_cast<autd::Controller*>(handle);
  wrapper->geometry().attenuation_coefficient() = attenuation;
}
bool AUTDGetFPGAInfo(void* const handle, uint8_t* out) {
  auto* const wrapper = static_cast<autd::Controller*>(handle);
  AUTD3_CAPI_TRY({
    const auto& res = wrapper->fpga_info();
    std::memcpy(out, &res[0], res.size());
    return true;
  })
}
int32_t AUTDUpdateCtrlFlags(void* const handle) {
  auto* const wrapper = static_cast<autd::Controller*>(handle);
  AUTD3_CAPI_TRY2(return wrapper->update_ctrl_flag() ? 1 : 0)
}

int32_t AUTDSetDelayOffset(void* const handle, const uint8_t* const delay, const uint8_t* const offset) {
  auto* const wrapper = static_cast<autd::Controller*>(handle);

  autd::DelayOffsets delay_offsets(wrapper->geometry().num_devices());

  if (delay != nullptr) {
    for (const auto& device : wrapper->geometry())
      for (const auto& transducer : device) delay_offsets[transducer.id()].delay = delay[transducer.id()];
  }
  if (offset != nullptr) {
    for (const auto& device : wrapper->geometry())
      for (const auto& transducer : device) delay_offsets[transducer.id()].offset = offset[transducer.id()];
  }

  AUTD3_CAPI_TRY2(return wrapper->send(delay_offsets) ? 1 : 0)
}

int32_t AUTDNumDevices(const void* const handle) {
  const auto* wrapper = static_cast<const autd::Controller*>(handle);
  const auto res = wrapper->geometry().num_devices();
  return static_cast<int32_t>(res);
}

void AUTDTransPosition(const void* const handle, const int32_t device_idx, const int32_t local_trans_idx, double* x, double* y, double* z) {
  const auto* wrapper = static_cast<const autd::Controller*>(handle);
  const auto& pos = wrapper->geometry()[device_idx][local_trans_idx].position();
  *x = pos.x();
  *y = pos.y();
  *z = pos.z();
}

void AUTDDeviceXDirection(const void* const handle, const int32_t device_idx, double* x, double* y, double* z) {
  const auto* wrapper = static_cast<const autd::Controller*>(handle);
  const auto& dir = wrapper->geometry()[device_idx].x_direction();
  *x = dir.x();
  *y = dir.y();
  *z = dir.z();
}
void AUTDDeviceYDirection(const void* const handle, const int32_t device_idx, double* x, double* y, double* z) {
  const auto* wrapper = static_cast<const autd::Controller*>(handle);
  const auto& dir = wrapper->geometry()[device_idx].y_direction();
  *x = dir.x();
  *y = dir.y();
  *z = dir.z();
}
void AUTDDeviceZDirection(const void* const handle, const int32_t device_idx, double* x, double* y, double* z) {
  const auto* wrapper = static_cast<const autd::Controller*>(handle);
  const auto& dir = wrapper->geometry()[device_idx].z_direction();
  *x = dir.x();
  *y = dir.y();
  *z = dir.z();
}

int32_t AUTDGetFirmwareInfoListPointer(void* const handle, void** out) {
  auto* const wrapper = static_cast<autd::Controller*>(handle);
  const auto size = static_cast<int32_t>(wrapper->geometry().num_devices());
  AUTD3_CAPI_TRY2({
    const auto res = wrapper->firmware_info_list();
    if (res.empty()) {
      last_error() = "filed to get some infos";
      return -1;
    }
    auto* list = firmware_info_list_create(res);
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
  firmware_info_list_delete(wrapper);
}

int32_t AUTDGetLastError(char* error) {
  const auto& error_ = last_error();
  const auto size = static_cast<int32_t>(error_.size() + 1);
  if (error == nullptr) return size;
  std::char_traits<char>::copy(error, error_.c_str(), size);
  return size;
}

void AUTDGainNull(void** gain) {
  auto* g = new autd::gain::Null;
  *gain = g;
}

void AUTDGainGrouped(void** gain) {
  auto* g = new autd::gain::Grouped;
  *gain = g;
}

void AUTDGainGroupedAdd(void* grouped_gain, const int32_t device_id, void* gain) {
  auto* const gg = dynamic_cast<autd::gain::Grouped*>(static_cast<autd::Gain*>(grouped_gain));
  auto* const g = static_cast<autd::Gain*>(gain);
  gg->add(device_id, std::shared_ptr<autd::Gain>(g));
}

void AUTDGainFocalPoint(void** gain, const double x, const double y, const double z, const uint8_t duty) {
  *gain = new autd::gain::FocalPoint(to_vec3(x, y, z), duty);
}
void AUTDGainBesselBeam(void** gain, const double x, const double y, const double z, const double n_x, const double n_y, const double n_z,
                        const double theta_z, const uint8_t duty) {
  *gain = new autd::gain::BesselBeam(to_vec3(x, y, z), to_vec3(n_x, n_y, n_z), theta_z, duty);
}
void AUTDGainPlaneWave(void** gain, const double n_x, const double n_y, const double n_z, const uint8_t duty) {
  *gain = new autd::gain::PlaneWave(to_vec3(n_x, n_y, n_z), duty);
}
void AUTDGainCustom(void** gain, const uint16_t* const data, const int32_t data_length) { *gain = new CustomGain(data, data_length); }
void AUTDGainTransducerTest(void** gain, const int32_t idx, const uint8_t duty, const uint8_t phase) {
  *gain = new autd::gain::TransducerTest(idx, duty, phase);
}
void AUTDDeleteGain(const void* const gain) {
  const auto* g = static_cast<const autd::Gain*>(gain);
  delete g;
}

void AUTDModulationStatic(void** mod, const uint8_t duty) { *mod = new autd::modulation::Static(duty); }
void AUTDModulationCustom(void** mod, const uint8_t* const buf, const uint32_t size, const uint32_t freq_div) {
  std::vector<uint8_t> buffer;
  for (uint32_t i = 0; i < size; i++) buffer.emplace_back(buf[i]);
  *mod = new CustomModulation(buffer, freq_div);
}

void AUTDModulationSquare(void** mod, const int32_t freq, const uint8_t low, const uint8_t high, const double duty) {
  *mod = new autd::modulation::Square(freq, low, high, duty);
}
void AUTDModulationSine(void** mod, const int32_t freq, const double amp, const double offset) {
  *mod = new autd::modulation::Sine(freq, amp, offset);
}
void AUTDModulationSineSquared(void** mod, const int32_t freq, const double amp, const double offset) {
  *mod = new autd::modulation::SineSquared(freq, amp, offset);
}
void AUTDModulationSineLegacy(void** mod, const double freq, const double amp, const double offset) {
  *mod = new autd::modulation::SineLegacy(freq, amp, offset);
}
uint32_t AUTDModulationSamplingFreqDiv(const void* const mod) {
  const auto* const m = static_cast<const autd::Modulation*>(mod);
  return static_cast<uint32_t>(m->sampling_freq_div_ratio());
}
void AUTDModulationSetSamplingFreqDiv(void* const mod, const uint32_t freq_div) {
  auto* const m = static_cast<autd::Modulation*>(mod);
  m->sampling_freq_div_ratio() = static_cast<size_t>(freq_div);
}
double AUTDModulationSamplingFreq(const void* const mod) {
  const auto* const m = static_cast<const autd::Modulation*>(mod);
  return m->sampling_freq();
}
void AUTDDeleteModulation(const void* const mod) {
  const auto* m = static_cast<const autd::Modulation*>(mod);
  delete m;
}

void AUTDSequence(void** out) { *out = new autd::sequence::PointSequence; }
void AUTDGainSequence(void** out, const uint16_t gain_mode) { *out = new autd::sequence::GainSequence(static_cast<autd::GAIN_MODE>(gain_mode)); }
bool AUTDSequenceAddPoint(void* const seq, const double x, const double y, const double z, const uint8_t duty) {
  auto* const seq_w = static_cast<autd::PointSequence*>(seq);
  AUTD3_CAPI_TRY({
    seq_w->add_point(to_vec3(x, y, z), duty);
    return true;
  })
}
bool AUTDSequenceAddPoints(void* const seq, const double* const points, const uint64_t points_size, const uint8_t* const duties,
                           const uint64_t duties_size) {
  auto* const seq_w = static_cast<autd::PointSequence*>(seq);
  std::vector<autd::Vector3> p;
  for (size_t i = 0; i < points_size; i++) p.emplace_back(to_vec3(points[3 * i], points[3 * i + 1], points[3 * i + 2]));

  std::vector<uint8_t> d;
  for (size_t i = 0; i < duties_size; i++) d.emplace_back(duties[i]);
  AUTD3_CAPI_TRY({
    seq_w->add_points(p, d);
    return true;
  })
}
bool AUTDSequenceAddGain(void* const seq, void* const gain) {
  auto* const seq_w = static_cast<autd::GainSequence*>(seq);
  auto* const g = static_cast<autd::Gain*>(gain);
  AUTD3_CAPI_TRY({
    seq_w->add_gain(std::shared_ptr<autd::Gain>(g));
    return true;
  })
}
double AUTDSequenceSetFreq(void* const seq, const double freq) {
  auto* const seq_w = static_cast<autd::core::Sequence*>(seq);
  return seq_w->set_frequency(freq);
}
double AUTDSequenceFreq(const void* const seq) {
  const auto* const seq_w = static_cast<const autd::core::Sequence*>(seq);
  return seq_w->frequency();
}
uint32_t AUTDSequencePeriod(const void* const seq) {
  const auto* const seq_w = static_cast<const autd::core::Sequence*>(seq);
  return static_cast<uint32_t>(seq_w->period_us());
}
uint32_t AUTDSequenceSamplingPeriod(const void* seq) {
  const auto* const seq_w = static_cast<const autd::core::Sequence*>(seq);
  return static_cast<uint32_t>(seq_w->sampling_period_us());
}
double AUTDSequenceSamplingFreq(const void* const seq) {
  const auto* const seq_w = static_cast<const autd::core::Sequence*>(seq);
  return seq_w->sampling_freq();
}
uint32_t AUTDSequenceSamplingFreqDiv(const void* const seq) {
  const auto* const seq_w = static_cast<const autd::core::Sequence*>(seq);
  return static_cast<uint32_t>(seq_w->sampling_freq_div_ratio());
}
void AUTDSequenceSetSamplingFreqDiv(void* const seq, const uint32_t freq_div) {
  auto* const seq_w = static_cast<autd::core::Sequence*>(seq);
  seq_w->sampling_freq_div_ratio() = static_cast<size_t>(freq_div);
}
void AUTDDeleteSequence(const void* const seq) {
  const auto* const seq_w = static_cast<const autd::core::Sequence*>(seq);
  delete seq_w;
}
void AUTDCircumSequence(void** out, const double x, const double y, const double z, const double nx, const double ny, const double nz,
                        const double radius, const uint64_t n) {
  *out = new autd::sequence::Circumference(to_vec3(x, y, z), to_vec3(nx, ny, nz), radius, n);
}

int32_t AUTDStop(void* const handle) {
  auto* const wrapper = static_cast<autd::Controller*>(handle);
  AUTD3_CAPI_TRY2(return wrapper->stop() ? 1 : 0)
}
int32_t AUTDPause(void* const handle) {
  auto* const wrapper = static_cast<autd::Controller*>(handle);
  AUTD3_CAPI_TRY2(return wrapper->pause() ? 1 : 0)
}
int32_t AUTDResume(void* const handle) {
  auto* const wrapper = static_cast<autd::Controller*>(handle);
  AUTD3_CAPI_TRY2(return wrapper->resume() ? 1 : 0)
}
int32_t AUTDSendHeader(void* const handle, void* const header) {
  auto* const wrapper = static_cast<autd::Controller*>(handle);
  auto* const h = static_cast<autd::core::IDatagramHeader*>(header);
  AUTD3_CAPI_TRY(return wrapper->send(*h) ? 1 : 0)
}
int32_t AUTDSendBody(void* const handle, void* const body) {
  auto* const wrapper = static_cast<autd::Controller*>(handle);
  auto* const b = static_cast<autd::core::IDatagramBody*>(body);
  AUTD3_CAPI_TRY(return wrapper->send(*b) ? 1 : 0)
}
int32_t AUTDSendHeaderBody(void* const handle, void* const header, void* const body) {
  auto* const wrapper = static_cast<autd::Controller*>(handle);
  auto* const h = static_cast<autd::core::IDatagramHeader*>(header);
  auto* const b = static_cast<autd::core::IDatagramBody*>(body);
  AUTD3_CAPI_TRY(return wrapper->send(*h, *b) ? 1 : 0)
}
void AUTDSTMController(void** out, void* handle) {
  auto* const wrapper = static_cast<autd::Controller*>(handle);
  *out = stm_controller_create(wrapper->stm());
}
bool AUTDAddSTMGain(const void* const handle, void* const gain) {
  auto* const wrapper = static_cast<const STMControllerWrapper*>(handle);
  auto* const g = static_cast<autd::Gain*>(gain);
  AUTD3_CAPI_TRY({
    wrapper->ptr->add_gain(*g);
    return true;
  })
}
bool AUTDStartSTM(const void* const handle, const double freq) {
  const auto* const wrapper = static_cast<const STMControllerWrapper*>(handle);
  AUTD3_CAPI_TRY({
    wrapper->ptr->start(freq);
    return true;
  })
}
bool AUTDStopSTM(const void* const handle) {
  const auto* const wrapper = static_cast<const STMControllerWrapper*>(handle);
  AUTD3_CAPI_TRY({
    wrapper->ptr->stop();
    return true;
  })
}
bool AUTDFinishSTM(const void* const handle) {
  const auto* const wrapper = static_cast<const STMControllerWrapper*>(handle);
  AUTD3_CAPI_TRY({
    wrapper->ptr->finish();
    stm_controller_delete(wrapper);
    return true;
  })
}
