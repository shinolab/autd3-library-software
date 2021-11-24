// File: logic.hpp
// Project: core
// Created Date: 11/05/2021
// Author: Shun Suzuki
// -----
// Last Modified: 24/11/2021
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
 * \brief Firmware logic
 */
class Logic {
 public:
  /**
   * \brief Get unique message id
   * \return message id
   */
  static uint8_t get_id() {
    static std::atomic id{MSG_NORMAL_BASE};

    if (uint8_t expected = 0xff; !id.compare_exchange_weak(expected, MSG_NORMAL_BASE)) id.fetch_add(0x01);

    return id.load();
  }

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
   * \brief Pack header with message id
   * \param msg_id message id
   * \param fpga_ctrl_flag fpga control flag
   * \param cpu_ctrl_flag cpu control flag
   * \param[out] data pointer to transmission data
   */
  static void pack_header(const uint8_t msg_id, const uint8_t fpga_ctrl_flag, const uint8_t cpu_ctrl_flag, uint8_t* const data) {
    auto* header = reinterpret_cast<GlobalHeader*>(data);
    header->msg_id = msg_id;
    header->fpga_ctrl_flags = fpga_ctrl_flag;
    header->cpu_ctrl_flags = cpu_ctrl_flag;
    header->mod_size = 0;
  }

  /**
   * \brief Pack header with modulation data
   * \param mod Modulation
   * \param fpga_ctrl_flag fpga control flag
   * \param cpu_ctrl_flag cpu control flag
   * \param[out] data pointer to transmission data
   * \param[out] mod_sent length of modulation data already sent
   * \return message id
   */
  static uint8_t pack_header(const ModulationPtr& mod, const uint8_t fpga_ctrl_flag, const uint8_t cpu_ctrl_flag, uint8_t* const data,
                             size_t* const mod_sent) {
    const uint8_t msg_id = get_id();
    pack_header(msg_id, fpga_ctrl_flag, cpu_ctrl_flag, data);
    if (mod == nullptr || *mod_sent == mod->buffer().size()) return msg_id;

    auto* header = reinterpret_cast<GlobalHeader*>(data);
    size_t offset = 0;
    if (*mod_sent == 0) {
      header->cpu_ctrl_flags |= MOD_BEGIN;
      const auto div = static_cast<uint16_t>(mod->sampling_freq_div_ratio() - 1);
      header->mod[0] = static_cast<uint8_t>(div & 0xFF);
      header->mod[1] = static_cast<uint8_t>(div >> 8 & 0xFF);
      offset += 2;
    }
    const auto mod_size = static_cast<uint8_t>(std::clamp(mod->buffer().size() - *mod_sent, size_t{0}, MOD_FRAME_SIZE - offset));
    if (*mod_sent + mod_size >= mod->buffer().size()) header->cpu_ctrl_flags |= MOD_END;
    header->mod_size = mod_size;

    std::memcpy(&header->mod[offset], &mod->buffer()[*mod_sent], mod_size);
    *mod_sent += mod_size;

    return msg_id;
  }

  /**
   * \brief Pack data body which contain phase and duty data of each transducer.
   * \param gain Gain
   * \param[out] data pointer to transmission data
   * \return size_t size to send
   * \details This function must be called after pack_header
   */
  static size_t pack_body(const GainPtr& gain, uint8_t* data) {
    if (gain == nullptr) return sizeof(GlobalHeader);

    auto* header = reinterpret_cast<GlobalHeader*>(data);
    header->cpu_ctrl_flags |= WRITE_BODY;

    auto* cursor = reinterpret_cast<uint16_t*>(data + sizeof(GlobalHeader));
    std::memcpy(cursor, gain->data().data(), gain->data().size() * sizeof(uint16_t));
    return sizeof(GlobalHeader) + gain->data().size() * sizeof(uint16_t);
  }

  /**
   * \brief Pack data body with sequence data
   * \param seq Sequence
   * \param geometry Geometry
   * \param[out] data pointer to transmission data
   * \param[out] seq_sent length of sequence data already sent
   * \return size_t size to send
   * \details This function must be called after pack_header
   */
  static size_t pack_body(const PointSequencePtr& seq, const Geometry& geometry, uint8_t* data, size_t* const seq_sent) {
    if (seq == nullptr || *seq_sent == seq->control_points().size()) return sizeof(GlobalHeader);

    auto* cursor = reinterpret_cast<uint16_t*>(data + sizeof(GlobalHeader));
    size_t offset = 1;
    auto* header = reinterpret_cast<GlobalHeader*>(data);
    header->cpu_ctrl_flags |= WRITE_BODY;
    if (*seq_sent == 0) {
      header->cpu_ctrl_flags |= SEQ_BEGIN;
      for (const auto& device : geometry) {
        cursor[device.id() * NUM_TRANS_IN_UNIT + 1] = static_cast<uint16_t>(seq->sampling_freq_div_ratio() - 1);
        cursor[device.id() * NUM_TRANS_IN_UNIT + 2] = static_cast<uint16_t>(geometry.wavelength() * 1000);
      }
      offset += 4;
    }
    const auto send_size =
        std::clamp(seq->control_points().size() - *seq_sent, size_t{0}, sizeof(uint16_t) * (NUM_TRANS_IN_UNIT - offset) / sizeof(SeqFocus));
    if (*seq_sent + send_size >= seq->control_points().size()) header->cpu_ctrl_flags |= SEQ_END;

    const auto fixed_num_unit = 256.0 / geometry.wavelength();
    for (const auto& device : geometry) {
      cursor[0] = static_cast<uint16_t>(send_size);
      auto* focus_cursor = reinterpret_cast<SeqFocus*>(&cursor[offset]);
      for (size_t i = *seq_sent; i < *seq_sent + send_size; i++, focus_cursor++) {
        const auto v = (device.to_local_position(seq->control_point(i)) * fixed_num_unit).cast<int32_t>();
        focus_cursor->set(v.x(), v.y(), v.z(), seq->duty(i));
      }
      cursor += NUM_TRANS_IN_UNIT;
    }
    *seq_sent += send_size;
    return sizeof(GlobalHeader) + geometry.num_transducers() * sizeof(uint16_t);
  }

  /**
   * \brief Pack data body with sequence data
   * \param seq Sequence
   * \param geometry Geometry
   * \param[out] data pointer to transmission data
   * \param[out] seq_sent length of sequence data already sent. Note that seq_sent should be (length of gain list) + 1 when all data is sent.
   * \return size_t size to send
   * \details This function must be called after pack_header
   */
  static size_t pack_body(const GainSequencePtr& seq, const Geometry& geometry, uint8_t* data, size_t* const seq_sent) {
    if (seq == nullptr || *seq_sent == seq->gains().size() + 1) return sizeof(GlobalHeader);

    auto* header = reinterpret_cast<GlobalHeader*>(data);
    header->cpu_ctrl_flags |= WRITE_BODY;
    const auto sent = static_cast<size_t>(seq->gain_mode());
    if (*seq_sent == 0) {
      header->cpu_ctrl_flags |= SEQ_BEGIN;
      auto* cursor = reinterpret_cast<uint16_t*>(data + sizeof(GlobalHeader));
      for (const auto& device : geometry) {
        cursor[device.id() * NUM_TRANS_IN_UNIT] = static_cast<uint16_t>(sent);
        cursor[device.id() * NUM_TRANS_IN_UNIT + 1] = static_cast<uint16_t>(seq->sampling_freq_div_ratio() - 1);
        cursor[device.id() * NUM_TRANS_IN_UNIT + 2] = static_cast<uint16_t>(seq->size());
      }
      *seq_sent += 1;
      return sizeof(GlobalHeader) + geometry.num_transducers() * sizeof(uint16_t);
    }

    if (*seq_sent + sent > seq->gains().size()) header->cpu_ctrl_flags |= SEQ_END;

    auto* cursor = reinterpret_cast<uint16_t*>(data + sizeof(GlobalHeader));
    const auto gain_idx = *seq_sent - 1;
    switch (seq->gain_mode()) {
      case GAIN_MODE::DUTY_PHASE_FULL:
        std::memcpy(cursor, seq->gains()[gain_idx]->data().data(), seq->gains()[gain_idx]->data().size() * sizeof(uint16_t));
        break;
      case GAIN_MODE::PHASE_FULL:
        for (const auto& dev : geometry)
          for (const auto& trans : dev)
            cursor[trans.id()] =
                core::utils::pack_to_u16(seq->gains()[gain_idx + 1]->data()[trans.id()].phase, seq->gains()[gain_idx]->data()[trans.id()].phase);
        break;
      case GAIN_MODE::PHASE_HALF:
        for (const auto& dev : geometry)
          for (const auto& trans : dev) {
            const auto phase1 = static_cast<uint8_t>(seq->gains()[gain_idx]->data()[trans.id()].phase >> 4 & 0x0F);
            const auto phase2 =
                static_cast<uint8_t>(gain_idx + 1 >= seq->size() ? 0x00 : seq->gains()[gain_idx + 1]->data()[trans.id()].phase & 0xF0);
            const auto phase3 =
                static_cast<uint8_t>(gain_idx + 2 >= seq->size() ? 0x00 : seq->gains()[gain_idx + 2]->data()[trans.id()].phase >> 4 & 0x0F);
            const auto phase4 =
                static_cast<uint8_t>(gain_idx + 3 >= seq->size() ? 0x00 : seq->gains()[gain_idx + 3]->data()[trans.id()].phase & 0xF0);
            cursor[trans.id()] = utils::pack_to_u16(phase4 | phase3, phase2 | phase1);
          }
        break;
    }
    *seq_sent += sent;
    return sizeof(GlobalHeader) + geometry.num_transducers() * sizeof(uint16_t);
  }

  /**
   * \brief Pack data body to set output delay and enable
   * \param delay_offset delay and offset data
   * \param[out] data pointer to transmission data
   * \return size_t size to send
   * \details This function must be called after pack_header
   */
  static size_t pack_delay_offset_body(const std::vector<DelayOffset>& delay_offset, uint8_t* data) {
    auto* header = reinterpret_cast<GlobalHeader*>(data);
    header->cpu_ctrl_flags |= WRITE_BODY;
    std::memcpy(data + sizeof(GlobalHeader), delay_offset.data(), delay_offset.size() * sizeof(uint16_t));
    return sizeof(GlobalHeader) + delay_offset.size() * sizeof(uint16_t);
  }
};
}  // namespace autd::core
