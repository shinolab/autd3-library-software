// File: interface.hpp
// Project: core
// Created Date: 12/12/2021
// Author: Shun Suzuki
// -----
// Last Modified: 12/12/2021
// Modified By: Shun Suzuki (suzuki@hapis.k.u-tokyo.ac.jp)
// -----
// Copyright (c) 2021 Hapis Lab. All rights reserved.
//

#pragma once

#include "geometry.hpp"
#include "hardware_defined.hpp"

namespace autd {
namespace core {

class IDatagram {
 public:
  IDatagram() = default;
  virtual ~IDatagram() = default;
  IDatagram(const IDatagram& v) noexcept = delete;
  IDatagram& operator=(const IDatagram& obj) = delete;
  IDatagram(IDatagram&& obj) = default;
  IDatagram& operator=(IDatagram&& obj) = default;

  virtual void init() = 0;
  virtual uint8_t pack(const Geometry& geometry, TxDatagram& tx, uint8_t& fpga_ctrl_flag, uint8_t& cpu_ctrl_flag) = 0;
  [[nodiscard]] virtual bool is_finished() const = 0;
};

class IDatagramBody : public IDatagram {
 public:
  IDatagramBody() : IDatagram() {}
};

class IDatagramHeader : public IDatagram {
 public:
  IDatagramHeader() : IDatagram() {}
};

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
static bool is_msg_processed(const size_t num_devices, const uint8_t msg_id, const RxDatagram& rx) {
  size_t processed = 0;
  for (size_t dev = 0; dev < num_devices; dev++)
    if (rx[dev].msg_id == msg_id) processed++;
  return processed == num_devices;
}

}  // namespace core
}  // namespace autd
