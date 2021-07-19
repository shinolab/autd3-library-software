// File: twincat_link.cpp
// Project: twincat
// Created Date: 08/03/2021
// Author: Shun Suzuki
// -----
// Last Modified: 19/07/2021
// Modified By: Shun Suzuki (suzuki@hapis.k.u-tokyo.ac.jp)
// -----
// Copyright (c) 2021 Hapis Lab. All rights reserved.
//

#include "autd3/link/emulator.hpp"

#if _WINDOWS
#include <WS2tcpip.h>
#else
#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <unistd.h>
#endif

#include "autd3/core/exception.hpp"
#include "autd3/core/geometry.hpp"
#include "autd3/core/link.hpp"

namespace autd::link {

class EmulatorImpl final : public Emulator {
 public:
  explicit EmulatorImpl(uint16_t port, const core::GeometryPtr& geometry);
  ~EmulatorImpl() override = default;
  EmulatorImpl(const EmulatorImpl& v) noexcept = delete;
  EmulatorImpl& operator=(const EmulatorImpl& obj) = delete;
  EmulatorImpl(EmulatorImpl&& obj) = delete;
  EmulatorImpl& operator=(EmulatorImpl&& obj) = delete;

  void open() override;
  void close() override;
  void send(const uint8_t* buf, size_t size) override;
  void read(uint8_t* rx, size_t buffer_len) override;
  bool is_open() override;

 private:
  bool _is_open;
  uint16_t _port;
#if _WINDOWS
  SOCKET _socket = {};
#else
  int _socket = 0;
#endif
  sockaddr_in _addr = {};

  core::COMMAND _last_command = core::COMMAND::OP;
  uint8_t _last_msg_id = 0;
  std::vector<uint8_t> _geometry_buf;

  static std::vector<uint8_t> init_geometry_buf(const core::GeometryPtr& geometry) {
    const auto vec_size = 9 * sizeof(float);
    const auto size = sizeof(core::RxGlobalHeader) + geometry->num_devices() * vec_size;
    std::vector<uint8_t> buf;
    buf.resize(size);

    auto* const uh = reinterpret_cast<core::RxGlobalHeader*>(&buf[0]);
    uh->msg_id = 0x00;
    uh->control_flags = 0x00;
    uh->command = core::COMMAND::EMULATOR_SET_GEOMETRY;
    uh->mod_size = 0x00;

    auto* const cursor = reinterpret_cast<float*>(&buf[sizeof(core::RxGlobalHeader)]);
    for (size_t i = 0; i < geometry->num_devices(); i++) {
      const auto trans_id = i * core::NUM_TRANS_IN_UNIT;
      auto origin = geometry->position(trans_id);
      auto right = geometry->x_direction(i);
      auto up = geometry->y_direction(i);
      cursor[9 * i] = static_cast<float>(origin.x());
      cursor[9 * i + 1] = static_cast<float>(origin.y());
      cursor[9 * i + 2] = static_cast<float>(origin.z());
      cursor[9 * i + 3] = static_cast<float>(right.x());
      cursor[9 * i + 4] = static_cast<float>(right.y());
      cursor[9 * i + 5] = static_cast<float>(right.z());
      cursor[9 * i + 6] = static_cast<float>(up.x());
      cursor[9 * i + 7] = static_cast<float>(up.y());
      cursor[9 * i + 8] = static_cast<float>(up.z());
    }

    return buf;
  }
};

core::LinkPtr Emulator::create(const uint16_t port, const core::GeometryPtr& geometry) {
  core::LinkPtr link = std::make_unique<EmulatorImpl>(port, geometry);
  return link;
}

EmulatorImpl::EmulatorImpl(const uint16_t port, const core::GeometryPtr& geometry)
    : _is_open(false), _port(port), _geometry_buf(init_geometry_buf(geometry)) {}

void EmulatorImpl::send(const uint8_t* buf, const size_t size) {
  const auto* header = reinterpret_cast<const core::RxGlobalHeader*>(buf);
  _last_msg_id = header->msg_id;
  _last_command = header->command;
  if (sendto(_socket, reinterpret_cast<const char*>(buf), static_cast<int>(size), 0, reinterpret_cast<sockaddr*>(&_addr), sizeof _addr) == -1)
    throw core::LinkError("failed to send data");
}

void EmulatorImpl::open() {
  if (this->is_open()) return;

#if _WINDOWS
#pragma warning(push)
#pragma warning(disable : 6031)
  WSAData wsa_data{};
  WSAStartup(MAKEWORD(2, 0), &wsa_data);
#pragma warning(pop)
#endif

  _socket = socket(AF_INET, SOCK_DGRAM, 0);
#if _WINDOWS
  if (_socket == INVALID_SOCKET)
#else
  if (_socket < 0)
#endif
    throw core::LinkError("cannot connect to emulator");

  _addr.sin_family = AF_INET;
  _addr.sin_port = htons(this->_port);
#if _WINDOWS
  const auto ip_addr("127.0.0.1");
  inet_pton(AF_INET, ip_addr, &_addr.sin_addr.S_un.S_addr);
#else
  _addr.sin_addr.s_addr = inet_addr("127.0.0.1");
#endif

  _is_open = true;
  this->send(&this->_geometry_buf[0], this->_geometry_buf.size());
}

void EmulatorImpl::close() {
  if (!this->is_open()) return;
#if _WINDOWS
  closesocket(_socket);
  WSACleanup();
#else
  ::close(_socket);
#endif
  _is_open = false;
}

void EmulatorImpl::read(uint8_t* rx, size_t buffer_len) {
  for (size_t i = 0; i < buffer_len; i += 2) rx[i + 1] = this->_last_msg_id;

  const auto set = [rx, buffer_len](const uint8_t value) {
    for (size_t i = 0; i < buffer_len; i += 2) rx[i] = value;
  };

  switch (this->_last_command) {
    case core::COMMAND::OP:
      break;
    case core::COMMAND::READ_CPU_VER_LSB:
    case core::COMMAND::READ_CPU_VER_MSB:
    case core::COMMAND::READ_FPGA_VER_LSB:
    case core::COMMAND::READ_FPGA_VER_MSB:
      set(0xFF);
      break;
    case core::COMMAND::SEQ_MODE:
    case core::COMMAND::CLEAR:
    case core::COMMAND::SET_DELAY_OFFSET:
    case core::COMMAND::PAUSE:
    case core::COMMAND::RESUME:
    case core::COMMAND::EMULATOR_SET_GEOMETRY:
      break;
  }
}

bool EmulatorImpl::is_open() { return _is_open; }

}  // namespace autd::link
