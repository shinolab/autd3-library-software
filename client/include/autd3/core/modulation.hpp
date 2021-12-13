// File: modulation.hpp
// Project: core
// Created Date: 11/05/2021
// Author: Shun Suzuki
// -----
// Last Modified: 13/12/2021
// Modified By: Shun Suzuki (suzuki@hapis.k.u-tokyo.ac.jp)
// -----
// Copyright (c) 2021 Hapis Lab. All rights reserved.
//

#pragma once

#include <vector>

#include "exception.hpp"
#include "hardware_defined.hpp"
#include "interface.hpp"
#include "logic.hpp"

namespace autd::core {

/**
 * @brief Modulation controls the amplitude modulation
 */
class Modulation : public IDatagramHeader {
 public:
  Modulation() noexcept : Modulation(10) {}
  explicit Modulation(const size_t freq_div) noexcept : _built(false), _freq_div_ratio(freq_div), _sent(0) {}
  ~Modulation() override = default;
  Modulation(const Modulation& v) noexcept = delete;
  Modulation& operator=(const Modulation& obj) = delete;
  Modulation(Modulation&& obj) = default;
  Modulation& operator=(Modulation&& obj) = default;

  /**
   * \brief Calculate modulation data
   */
  virtual void calc() = 0;

  /**
   * \brief Build modulation data
   */
  void build() {
    if (this->_built) return;
    if (_freq_div_ratio > MOD_SAMPLING_FREQ_DIV_MAX)
      throw exception::ModulationBuildError("Modulation sampling frequency division ratio is out of range");
    this->calc();
    if (this->_buffer.size() > MOD_BUF_SIZE_MAX) throw exception::ModulationBuildError("Modulation buffer overflow");
    this->_built = true;
  }

  /**
   * \brief Re-build modulation data
   */
  void rebuild() {
    this->_built = false;
    this->build();
  }

  /**
   * \brief modulation data
   */
  [[nodiscard]] const std::vector<uint8_t>& buffer() const { return _buffer; }

  /**
   * \brief sampling frequency division ratio
   * \details sampling frequency will be autd::core::MOD_SAMPLING_FREQ_BASE /(sampling frequency division ratio). The value must be in 1, 2, ...,
   * autd::core::MOD_SAMPLING_FREQ_DIV_MAX.
   */
  size_t& sampling_freq_div_ratio() noexcept { return _freq_div_ratio; }

  /**
   * \brief sampling frequency division ratio
   * \details sampling frequency will be autd::core::MOD_SAMPLING_FREQ_BASE /(sampling frequency division ratio). The value must be in 1, 2, ...,
   * autd::core::MOD_SAMPLING_FREQ_DIV_MAX.
   */
  [[nodiscard]] size_t sampling_freq_div_ratio() const noexcept { return _freq_div_ratio; }

  /**
   * \brief modulation sampling frequency
   */
  [[nodiscard]] double sampling_freq() const noexcept { return static_cast<double>(MOD_SAMPLING_FREQ_BASE) / static_cast<double>(_freq_div_ratio); }

  void init() override {
    this->build();
    _sent = 0;
  }

  uint8_t pack(TxDatagram& tx, const uint8_t fpga_ctrl_flag, const uint8_t cpu_ctrl_flag) override {
    const uint8_t msg_id = get_id();

    auto* header = reinterpret_cast<GlobalHeader*>(tx.data());
    header->msg_id = msg_id;
    header->fpga_ctrl_flags = fpga_ctrl_flag;
    header->cpu_ctrl_flags = cpu_ctrl_flag;
    header->mod_size = 0;

    tx.num_bodies() = 0;

    if (is_finished()) return msg_id;

    size_t offset = 0;
    if (_sent == 0) {
      header->cpu_ctrl_flags |= MOD_BEGIN;
      const auto div = static_cast<uint16_t>(sampling_freq_div_ratio() - 1);
      header->mod[0] = static_cast<uint8_t>(div & 0xFF);
      header->mod[1] = static_cast<uint8_t>(div >> 8 & 0xFF);
      offset += 2;
    }
    const auto mod_size = static_cast<uint8_t>(std::clamp(_buffer.size() - _sent, size_t{0}, MOD_FRAME_SIZE - offset));
    if (_sent + mod_size >= _buffer.size()) header->cpu_ctrl_flags |= MOD_END;
    header->mod_size = mod_size;

    std::memcpy(&header->mod[offset], &_buffer[_sent], mod_size);
    _sent += mod_size;

    return msg_id;
  }

  [[nodiscard]] bool is_finished() const override { return _sent == _buffer.size(); }

 protected:
  bool _built;
  size_t _freq_div_ratio;
  std::vector<uint8_t> _buffer;
  size_t _sent;
};

}  // namespace autd::core
