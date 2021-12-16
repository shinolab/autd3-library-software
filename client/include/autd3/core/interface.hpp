// File: interface.hpp
// Project: core
// Created Date: 12/12/2021
// Author: Shun Suzuki
// -----
// Last Modified: 15/12/2021
// Modified By: Shun Suzuki (suzuki@hapis.k.u-tokyo.ac.jp)
// -----
// Copyright (c) 2021 Hapis Lab. All rights reserved.
//

#pragma once

#include "geometry.hpp"
#include "hardware_defined.hpp"

namespace autd {
namespace core {
namespace datagram {

class IDatagramHeader {
 public:
  IDatagramHeader() = default;
  virtual ~IDatagramHeader() = default;
  IDatagramHeader(const IDatagramHeader& v) noexcept = delete;
  IDatagramHeader& operator=(const IDatagramHeader& obj) = delete;
  IDatagramHeader(IDatagramHeader&& obj) = default;
  IDatagramHeader& operator=(IDatagramHeader&& obj) = default;

  virtual void init() = 0;
  virtual void pack(uint8_t msg_id, TxDatagram& tx, uint8_t fpga_ctrl_flag, uint8_t cpu_ctrl_flag) = 0;
  [[nodiscard]] virtual bool is_finished() const = 0;
};

class IDatagramBody {
 public:
  IDatagramBody() = default;
  virtual ~IDatagramBody() = default;
  IDatagramBody(const IDatagramBody& v) noexcept = delete;
  IDatagramBody& operator=(const IDatagramBody& obj) = delete;
  IDatagramBody(IDatagramBody&& obj) = default;
  IDatagramBody& operator=(IDatagramBody&& obj) = default;

  virtual void init() = 0;
  virtual void pack(const Geometry& geometry, TxDatagram& tx) = 0;
  [[nodiscard]] virtual bool is_finished() const = 0;
};

}  // namespace datagram
}  // namespace core
}  // namespace autd
