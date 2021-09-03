// File: logic.hpp
// Project: core
// Created Date: 11/05/2021
// Author: Shun Suzuki
// -----
// Last Modified: 03/09/2021
// Modified By: Shun Suzuki (suzuki@hapis.k.u-tokyo.ac.jp)
// -----
// Copyright (c) 2021 Hapis Lab. All rights reserved.
//

#pragma once

#include <atomic>
#include <memory>
#include <thread>
#include <vector>

#include "gain.hpp"
#include "geometry.hpp"
#include "modulation.hpp"
#include "sequence.hpp"

namespace autd::core {
/**
 * \brief Hardware logic
 */
class Logic {
  static uint8_t get_id() {
    static std::atomic<uint8_t> id{0};

    id.fetch_add(0x01);
    uint8_t expected = 0xff;
    id.compare_exchange_weak(expected, 1);

    return id.load();
  }

 public:
  /**
   * \brief check if the data which have msg_id have been processed in the devices.
   * \param num_devices number of devices
   * \param msg_id message id
   * \param rx pointer to received data
   * \return whether the data have been processed
   */
  static bool is_msg_processed(const size_t num_devices, const uint8_t msg_id, const uint8_t* const rx) {
    size_t processed = 0;
    for (size_t dev = 0; dev < num_devices; dev++)
      if (const uint8_t proc_id = rx[dev * 2 + 1]; proc_id == msg_id) processed++;
    return processed == num_devices;
  }

  /**
   * \brief Pack header with COMMAND
   * \param cmd command
   * \param ctrl_flag control flag
   * \param[out] data pointer to transmission data
   * \param[out] msg_id message id
   */
  static void pack_header(const COMMAND cmd, const uint8_t ctrl_flag, uint8_t* data, uint8_t* const msg_id) {
    auto* header = reinterpret_cast<RxGlobalHeader*>(data);
    *msg_id = get_id();
    header->msg_id = *msg_id;
    header->control_flags = ctrl_flag;
    header->mod_size = 0;
    header->command = cmd;
  }

  /**
   * \brief Pack header with modulation data
   * \param mod Modulation
   * \param ctrl_flag control flag
   * \param[out] data pointer to transmission data
   * \param[out] msg_id message id
   */
  static void pack_header(const ModulationPtr& mod, const uint8_t ctrl_flag, uint8_t* data, uint8_t* const msg_id) {
    pack_header(COMMAND::OP, ctrl_flag, data, msg_id);
    if (mod == nullptr) return;
    auto* header = reinterpret_cast<RxGlobalHeader*>(data);
    size_t offset = 0;
    if (mod->sent() == 0) {
      header->control_flags |= MOD_BEGIN;
      header->mod[0] = static_cast<uint8_t>(mod->sampling_frequency_division() & 0xFF);
      header->mod[1] = static_cast<uint8_t>(mod->sampling_frequency_division() >> 8 & 0xFF);
      offset += 2;
    }
    const auto mod_size = static_cast<uint8_t>(std::clamp(mod->buffer().size() - mod->sent(), size_t{0}, MOD_FRAME_SIZE - offset));
    if (mod->sent() + mod_size >= mod->buffer().size()) header->control_flags |= MOD_END;
    header->mod_size = mod_size;

    std::memcpy(&header->mod[offset], &mod->buffer()[mod->sent()], mod_size);
    mod->sent() += mod_size;
  }

  /**
   * \brief Pack data body which contain phase and duty data of each transducer.
   * \param gain Gain
   * \param[out] data pointer to transmission data
   * \param[out] size size to send
   */
  static void pack_body(const GainPtr& gain, uint8_t* data, size_t* size) {
    const auto num_devices = gain != nullptr ? gain->data().size() : 0;

    *size = sizeof(RxGlobalHeader) + sizeof(uint16_t) * NUM_TRANS_IN_UNIT * num_devices;
    if (gain == nullptr) return;

    auto* cursor = data + sizeof(RxGlobalHeader);
    const auto byte_size = NUM_TRANS_IN_UNIT * sizeof(uint16_t);
    for (size_t i = 0; i < num_devices; i++, cursor += byte_size) std::memcpy(cursor, &gain->data()[i].at(0), byte_size);
  }

  /**
   * \brief Pack data body with sequence data
   * \param seq Sequence
   * \param geometry Geometry
   * \param[out] data pointer to transmission data
   * \param[out] size size to send
   */
  static void pack_body(const PointSequencePtr& seq, const GeometryPtr& geometry, uint8_t* data, size_t* const size) {
    const auto num_devices = seq != nullptr ? geometry->num_devices() : 0;

    *size = sizeof(RxGlobalHeader) + sizeof(uint16_t) * NUM_TRANS_IN_UNIT * num_devices;
    if (seq == nullptr) return;

    auto* cursor = reinterpret_cast<uint16_t*>(data + sizeof(RxGlobalHeader));
    size_t offset = 1;
    auto* header = reinterpret_cast<RxGlobalHeader*>(data);
    if (seq->sent() == 0) {
      header->control_flags |= SEQ_BEGIN;
      for (size_t device = 0; device < num_devices; device++) {
        cursor[device * NUM_TRANS_IN_UNIT + 1] = seq->sampling_frequency_division();
        cursor[device * NUM_TRANS_IN_UNIT + 2] = static_cast<uint16_t>(geometry->wavelength() * 1000);
      }
      offset += 4;
    }
    const auto send_size = static_cast<uint16_t>(
        std::clamp(seq->control_points().size() - seq->sent(), size_t{0}, sizeof(uint16_t) * (NUM_TRANS_IN_UNIT - offset) / sizeof(SeqFocus)));
    if (seq->sent() + send_size >= seq->control_points().size()) header->control_flags |= SEQ_END;

    const auto fixed_num_unit = 256.0 / geometry->wavelength();
    for (size_t device = 0; device < num_devices; device++, cursor += NUM_TRANS_IN_UNIT) {
      cursor[0] = send_size;
      auto* focus_cursor = reinterpret_cast<SeqFocus*>(&cursor[offset]);
      for (size_t i = seq->sent(); i < seq->sent() + send_size; i++, focus_cursor++) {
        const auto v = (geometry->to_local_position(device, seq->control_point(i)) * fixed_num_unit).cast<int32_t>();
        focus_cursor->set(v.x(), v.y(), v.z(), seq->duty(i));
      }
    }
    seq->sent() += send_size;
  }

  /**
   * \brief Pack data body with sequence data
   * \param seq Sequence
   * \param geometry Geometry
   * \param[out] data pointer to transmission data
   * \param[out] size size to send
   */
  static void pack_body(const GainSequencePtr& seq, const GeometryPtr& geometry, uint8_t* data, size_t* const size) {
    const auto num_devices = seq != nullptr ? geometry->num_devices() : 0;

    *size = sizeof(RxGlobalHeader) + sizeof(uint16_t) * NUM_TRANS_IN_UNIT * num_devices;
    if (seq == nullptr) return;

    auto* header = reinterpret_cast<RxGlobalHeader*>(data);
    const auto seq_sent = static_cast<size_t>(seq->gain_mode());
    if (seq->sent() == 0) {
      header->control_flags |= SEQ_BEGIN;
      auto* cursor = reinterpret_cast<uint16_t*>(data + sizeof(RxGlobalHeader));
      for (size_t device = 0; device < num_devices; device++) {
        cursor[device * NUM_TRANS_IN_UNIT] = static_cast<uint16_t>(seq_sent);
        cursor[device * NUM_TRANS_IN_UNIT + 1] = seq->sampling_frequency_division();
        cursor[device * NUM_TRANS_IN_UNIT + 2] = static_cast<uint16_t>(seq->size());
      }
      seq->sent()++;
      return;
    }

    if (seq->sent() + seq_sent > seq->gains().size()) header->control_flags |= SEQ_END;

    auto* cursor = data + sizeof(RxGlobalHeader);
    const auto byte_size = NUM_TRANS_IN_UNIT * sizeof(uint16_t);
    const auto gain_idx = seq->sent() - 1;
    for (size_t device = 0; device < num_devices; device++, cursor += byte_size) {
      switch (seq->gain_mode()) {
        case GAIN_MODE::DUTY_PHASE_FULL:
          std::memcpy(cursor, &seq->gains()[gain_idx]->data()[device].at(0), byte_size);
          break;
        case GAIN_MODE::PHASE_FULL:
          for (size_t i = 0; i < NUM_TRANS_IN_UNIT; i++) {
            cursor[2 * i] = static_cast<uint8_t>(seq->gains()[gain_idx]->data()[device].at(i) & 0xFF);
            cursor[2 * i + 1] = static_cast<uint8_t>(gain_idx + 1 >= seq->size() ? 0x00 : seq->gains()[gain_idx + 1]->data()[device].at(i) & 0xFF);
          }
          break;
        case GAIN_MODE::PHASE_HALF:
          for (size_t i = 0; i < NUM_TRANS_IN_UNIT; i++) {
            const auto phase1 = static_cast<uint8_t>((seq->gains()[gain_idx]->data()[device].at(i) >> 4) & 0xF);
            const auto phase2 =
                static_cast<uint8_t>(gain_idx + 1 >= seq->size() ? 0x00 : (seq->gains()[gain_idx + 1]->data()[device].at(i) >> 4) & 0xF);
            const auto phase3 =
                static_cast<uint8_t>(gain_idx + 2 >= seq->size() ? 0x00 : (seq->gains()[gain_idx + 2]->data()[device].at(i) >> 4) & 0xF);
            const auto phase4 =
                static_cast<uint8_t>(gain_idx + 3 >= seq->size() ? 0x00 : (seq->gains()[gain_idx + 3]->data()[device].at(i) >> 4) & 0xF);
            cursor[2 * i] = (phase2 << 4) | phase1;
            cursor[2 * i + 1] = (phase4 << 4) | phase3;
          }
          break;
      }
    }
    seq->sent() += seq_sent;
  }

  /**
   * \brief Pack data body to set output delay and enable
   * \param delay delay data of each transducer
   * \param offset duty offset data of each transducer
   * \param[out] data pointer to transmission data
   * \param[out] size size to send
   */
  static void pack_delay_offset_body(const std::vector<std::array<uint8_t, NUM_TRANS_IN_UNIT>>& delay,
                                     const std::vector<std::array<uint8_t, NUM_TRANS_IN_UNIT>>& offset, uint8_t* data, size_t* const size) {
    *size = sizeof(RxGlobalHeader) + sizeof(uint16_t) * NUM_TRANS_IN_UNIT * delay.size();
    auto* cursor = reinterpret_cast<uint16_t*>(data + sizeof(RxGlobalHeader));
    for (size_t dev = 0; dev < delay.size(); dev++)
      for (size_t i = 0; i < NUM_TRANS_IN_UNIT; i++) *cursor++ = Utilities::pack_to_u16(offset[dev][i], delay[dev][i]);
  }
};
}  // namespace autd::core
