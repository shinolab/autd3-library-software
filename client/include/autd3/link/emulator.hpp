// File: emulator.hpp
// Project: link
// Created Date: 05/07/2021
// Author: Shun Suzuki
// -----
// Last Modified: 12/12/2021
// Modified By: Shun Suzuki (suzuki@hapis.k.u-tokyo.ac.jp)
// -----
// Copyright (c) 2021 Hapis Lab. All rights reserved.
//

#pragma once

#include <memory>

#include "autd3/core/geometry.hpp"
#include "autd3/core/link.hpp"

namespace autd::link {

/**
 * \brief Emulator link for [AUTD-emulator](https://github.com/shinolab/autd-emulator)
 */
class Emulator : virtual public core::Link {
 public:
  /**
   * @brief Create Emulator link
   * @param port port
   * @param geometry geometry
   */
  static core::LinkPtr create(uint16_t port, const core::Geometry& geometry);

  Emulator() = default;
  ~Emulator() override = default;
  Emulator(const Emulator& v) noexcept = delete;
  Emulator& operator=(const Emulator& obj) = delete;
  Emulator(Emulator&& obj) = delete;
  Emulator& operator=(Emulator&& obj) = delete;

  void open() override = 0;
  void close() override = 0;
  void send(const core::TxDatagram& tx) override = 0;
  void receive(core::RxDatagram& rx) override = 0;
  bool is_open() override = 0;
};
}  // namespace autd::link
