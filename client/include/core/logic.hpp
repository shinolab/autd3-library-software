// File: logic.hpp
// Project: core
// Created Date: 11/05/2021
// Author: Shun Suzuki
// -----
// Last Modified: 02/06/2021
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
    const auto mod_size = static_cast<uint8_t>(std::clamp(mod->buffer().size() - mod->sent(), size_t{0}, MOD_FRAME_SIZE));
    header->mod_size = mod_size;
    if (mod->sent() == 0) header->control_flags |= MOD_BEGIN;
    if (mod->sent() + mod_size >= mod->buffer().size()) header->control_flags |= MOD_END;

    std::memcpy(header->mod, &mod->buffer()[mod->sent()], mod_size);
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
    for (size_t i = 0; i < num_devices; i++) {
      std::memcpy(cursor, &gain->data()[i].at(0), byte_size);
      cursor += byte_size;
    }
  }

  /**
   * \brief Pack data body with sequence data
   * \param seq Sequence
   * \param geometry Geometry
   * \param[out] data pointer to transmission data
   * \param[out] size size to send
   */
  static void pack_body(const SequencePtr& seq, const GeometryPtr& geometry, uint8_t* data, size_t* const size) {
    const auto num_devices = seq != nullptr ? geometry->num_devices() : 0;

    *size = sizeof(RxGlobalHeader) + sizeof(uint16_t) * NUM_TRANS_IN_UNIT * num_devices;
    if (seq == nullptr) return;

    const auto send_size = static_cast<uint16_t>(std::clamp(seq->control_points().size() - seq->sent(), size_t{0}, size_t{49}));

    auto* header = reinterpret_cast<RxGlobalHeader*>(data);
    if (seq->sent() == 0) header->control_flags |= SEQ_BEGIN;
    if (seq->sent() + send_size >= seq->control_points().size()) header->control_flags |= SEQ_END;

    auto* cursor = reinterpret_cast<uint16_t*>(data + sizeof(RxGlobalHeader));
    const auto fixed_num_unit = 255 / geometry->wavelength();
    for (size_t device = 0; device < num_devices; device++) {
      cursor[0] = send_size;
      cursor[1] = seq->sampling_frequency_division();
      cursor[2] = static_cast<uint16_t>(geometry->wavelength() * 1000);
      auto* focus_cursor = reinterpret_cast<SeqFocus*>(&cursor[4]);
      for (size_t i = 0; i < send_size; i++) {
        auto v64 = geometry->local_position(device, seq->control_points()[seq->sent() + i]);
        const auto x = static_cast<uint32_t>(static_cast<int32_t>(v64.x() * fixed_num_unit));
        const auto y = static_cast<uint32_t>(static_cast<int32_t>(v64.y() * fixed_num_unit));
        const auto z = static_cast<uint32_t>(static_cast<int32_t>(v64.z() * fixed_num_unit));
        focus_cursor->set(x, y, z, 0xFF);
        focus_cursor++;
      }
      cursor += NUM_TRANS_IN_UNIT;
    }
    seq->sent() += send_size;
  }

  /**
   * \brief Pack data body to synchronize devices
   * \param config Configuration for Modulation
   * \param num_devices number of devices
   * \param[out] data pointer to transmission data
   * \param[out] size size to send
   */
  static void pack_sync_body(const Configuration config, const size_t num_devices, uint8_t* data, size_t* const size) {
    *size = sizeof(RxGlobalHeader) + sizeof(uint16_t) * NUM_TRANS_IN_UNIT * num_devices;

    auto* cursor = reinterpret_cast<uint16_t*>(data + sizeof(RxGlobalHeader));
    for (size_t i = 0; i < num_devices; i++) {
      cursor[0] = config.mod_buf_size();
      cursor[1] = config.mod_sampling_freq_div();
      cursor += NUM_TRANS_IN_UNIT;
    }
  }

  /**
   * \brief Pack data body to set output delay
   * \param delay delay data of each transducer
   * \param[out] data pointer to transmission data
   * \param[out] size size to send
   */
  static void pack_delay_body(const std::vector<DataArray>& delay, uint8_t* data, size_t* const size) {
    *size = sizeof(RxGlobalHeader) + sizeof(uint16_t) * NUM_TRANS_IN_UNIT * delay.size();
    auto* cursor = reinterpret_cast<uint16_t*>(data + sizeof(RxGlobalHeader));
    for (auto&& d : delay) {
      std::memcpy(cursor, &d[0], NUM_TRANS_IN_UNIT);
      cursor += NUM_TRANS_IN_UNIT;
    }
  }
};
}  // namespace autd::core
