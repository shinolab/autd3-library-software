// File: logic.hpp
// Project: core
// Created Date: 11/05/2021
// Author: Shun Suzuki
// -----
// Last Modified: 14/05/2021
// Modified By: Shun Suzuki (suzuki@hapis.k.u-tokyo.ac.jp)
// -----
// Copyright (c) 2021 Hapis Lab. All rights reserved.
//

#pragma once

#include <atomic>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include "gain.hpp"
#include "geometry.hpp"
#include "modulation.hpp"
#include "result.hpp"
#include "sequence.hpp"

namespace autd::core {
class Logic {
 public:
  static uint8_t get_id() {
    static std::atomic<uint8_t> id{0};

    id.fetch_add(0x01);
    uint8_t expected = 0xff;
    id.compare_exchange_weak(expected, 0);

    return id.load();
  }

  static void PackHeader(const COMMAND cmd, const bool silent_mode, const bool seq_mode, uint8_t* data, uint8_t* const msg_id) {
    auto* header = reinterpret_cast<RxGlobalHeader*>(data);
    *msg_id = get_id();
    header->msg_id = *msg_id;
    header->control_flags = 0;
    header->mod_size = 0;
    header->command = cmd;

    if (seq_mode) header->control_flags |= SEQ_MODE;
    if (silent_mode) header->control_flags |= SILENT;
  }

  static void PackHeader(const ModulationPtr& mod, const bool silent_mode, const bool seq_mode, uint8_t* data, uint8_t* const msg_id) {
    PackHeader(COMMAND::OP, silent_mode, seq_mode, data, msg_id);
    if (mod == nullptr) return;
    auto* header = reinterpret_cast<RxGlobalHeader*>(data);
    const auto mod_size = static_cast<uint8_t>(std::clamp(mod->buffer().size() - mod->sent(), size_t{0}, MOD_FRAME_SIZE));
    header->mod_size = mod_size;
    if (mod->sent() == 0) header->control_flags |= MOD_BEGIN;
    if (mod->sent() + mod_size >= mod->buffer().size()) header->control_flags |= MOD_END;

    std::memcpy(header->mod, &mod->buffer()[mod->sent()], mod_size);
    mod->sent() += mod_size;
  }

  static void PackBody(const GainPtr& gain, uint8_t* data, size_t* size) {
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

  void static PackBody(const SequencePtr seq, const GeometryPtr geometry, uint8_t* data, size_t* const size) {
    const auto num_devices = seq != nullptr ? geometry->num_devices() : 0;

    *size = sizeof(RxGlobalHeader) + sizeof(uint16_t) * NUM_TRANS_IN_UNIT * num_devices;
    if (seq == nullptr) return;

    const auto send_size = static_cast<uint16_t>(std::clamp(seq->control_points().size() - seq->sent(), size_t{0}, size_t{49}));

    auto* header = reinterpret_cast<RxGlobalHeader*>(data);
    if (seq->sent() == 0) header->control_flags |= SEQ_BEGIN;
    if (seq->sent() + send_size >= seq->control_points().size()) header->control_flags |= SEQ_END;

    auto* cursor = reinterpret_cast<uint16_t*>(data + sizeof(RxGlobalHeader));
    const auto fixed_num_unit = geometry->wavelength() / 255;
    for (size_t device = 0; device < num_devices; device++) {
      cursor[0] = send_size;
      cursor[1] = seq->sampling_frequency_division();
      cursor[2] = static_cast<uint16_t>(geometry->wavelength() * 1000);
      auto* focus_cursor = reinterpret_cast<SeqFocus*>(&cursor[4]);
      for (size_t i = 0; i < send_size; i++) {
        auto v64 = geometry->local_position(device, seq->control_points()[seq->sent() + i]);
        const auto x = static_cast<uint32_t>(static_cast<int32_t>(v64.x() / fixed_num_unit));
        const auto y = static_cast<uint32_t>(static_cast<int32_t>(v64.y() / fixed_num_unit));
        const auto z = static_cast<uint32_t>(static_cast<int32_t>(v64.z() / fixed_num_unit));
        SeqFocus focus;
        focus.x15_0 = x & 0xFFFF;
        focus.y7_0_x23_16 = ((y << 8) & 0xFF00) | ((x >> 24) & 0x80) | ((x >> 16) & 0x7F);
        focus.y23_8 = ((x >> 16) & 0x8000) | ((x >> 8) & 0x7FFF);
        focus.z15_0 = z & 0xFFFF;
        focus.duty_z23_16 = 0xFF00 | ((z >> 24) & 0x80) | ((z >> 16) & 0x7F);  // duty = 0xFF
        *focus_cursor = focus;
        focus_cursor++;
      }
      cursor += NUM_TRANS_IN_UNIT;
    }
    seq->sent() += send_size;
  }

  [[nodiscard]] static Result<bool, std::string> PackSyncBody(const Configuration config, const size_t num_devices, uint8_t* data,
                                                              size_t* const size) {
    *size = sizeof(RxGlobalHeader) + sizeof(uint16_t) * NUM_TRANS_IN_UNIT * num_devices;

    const auto mod_sampling_freq = static_cast<uint32_t>(config.mod_sampling_freq());
    const auto mod_buf_size = static_cast<uint32_t>(config.mod_buf_size());

    if (mod_buf_size < mod_sampling_freq) return Err(std::string("Modulation buffer size must be not less than sampling frequency"));

    const auto mod_idx_shift = Log2U(MOD_SAMPLING_FREQ_BASE / mod_sampling_freq);
    const auto ref_clk_cyc_shift = Log2U(mod_buf_size / mod_sampling_freq);

    auto* cursor = reinterpret_cast<uint16_t*>(data + sizeof(RxGlobalHeader));
    for (size_t i = 0; i < num_devices; i++) {
      cursor[0] = mod_idx_shift;
      cursor[1] = ref_clk_cyc_shift;
      cursor += NUM_TRANS_IN_UNIT;
    }

    return Ok(true);
  }

 private:
  static uint16_t Log2U(const uint32_t x) {
#ifdef _MSC_VER
    unsigned long n;         // NOLINT
    _BitScanReverse(&n, x);  // NOLINT
#else
    uint32_t n;
    n = 31 - __builtin_clz(x);
#endif
    return static_cast<uint16_t>(n);
  }
};
}  // namespace autd::core
