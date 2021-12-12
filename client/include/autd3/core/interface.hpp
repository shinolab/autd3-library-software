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

class TxDatagram final {
 public:
  TxDatagram() : _data(nullptr), _header_size(0), _body_size(0), _num_bodies(0) {}

  explicit TxDatagram(const size_t device_num) : _header_size(sizeof(GlobalHeader)), _body_size(sizeof(Body)), _num_bodies(0) {
    _data = std::make_unique<uint8_t[]>(_header_size + device_num * _body_size);
  }

  [[nodiscard]] const uint8_t* data() const { return _data.get(); }
  [[nodiscard]] const uint8_t* header() const { return _data.get(); }
  [[nodiscard]] const uint8_t* body(size_t i) const { return _data.get() + _header_size + _body_size * i; }

  uint8_t* data() { return _data.get(); }
  uint8_t* header() { return _data.get(); }
  uint8_t* body(const size_t i) { return _data.get() + _header_size + _body_size * i; }

  [[nodiscard]] size_t header_size() const { return _header_size; }
  [[nodiscard]] size_t body_size() const { return _body_size; }
  [[nodiscard]] size_t num_bodies() const { return _num_bodies; }
  size_t& num_bodies() { return _num_bodies; }

  void copy_from(const TxDatagram& other) {
    _header_size = other.header_size();
    _body_size = other.body_size();
    _num_bodies = other.num_bodies();
    std::memcpy(_data.get(), other.data(), _header_size + _num_bodies * _body_size);
  }

 private:
  std::unique_ptr<uint8_t[]> _data;
  size_t _header_size;
  size_t _body_size;
  size_t _num_bodies;
};

class RxDatagram final {
 public:
  RxDatagram() : _data(nullptr) {}

  explicit RxDatagram(const size_t device_num) { _data = std::make_unique<RxMessage[]>(device_num); }

  [[nodiscard]] const uint8_t* data() const { return reinterpret_cast<const uint8_t*>(_data.get()); }

  RxMessage const& operator[](const size_t i) const { return _data[i]; }

 private:
  std::unique_ptr<RxMessage[]> _data;
};

class IDatagram {
 public:
  IDatagram() = default;
  virtual ~IDatagram() = default;
  IDatagram(const IDatagram& v) noexcept = delete;
  IDatagram& operator=(const IDatagram& obj) = delete;
  IDatagram(IDatagram&& obj) = default;
  IDatagram& operator=(IDatagram&& obj) = default;

  virtual void init() = 0;
  virtual uint8_t pack(const Geometry& geometry, TxDatagram& tx, uint8_t&& fpga_ctrl_flag, uint8_t&& cpu_ctrl_flag);
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
