// File: twincat_link.cpp
// Project: twincat
// Created Date: 08/03/2021
// Author: Shun Suzuki
// -----
// Last Modified: 22/11/2021
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
  explicit EmulatorImpl(uint16_t port, const core::Geometry& geometry);
  ~EmulatorImpl() override = default;
  EmulatorImpl(const EmulatorImpl& v) noexcept = delete;
  EmulatorImpl& operator=(const EmulatorImpl& obj) = delete;
  EmulatorImpl(EmulatorImpl&& obj) = delete;
  EmulatorImpl& operator=(EmulatorImpl&& obj) = delete;

  void open() override;
  void close() override;
  void send(const uint8_t* buf, size_t size) override;
  void receive(uint8_t* rx, size_t buffer_len) override;
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

  uint8_t _last_msg_id = 0;
  std::vector<uint8_t> _geometry_buf;

  static std::vector<uint8_t> init_geometry_buf(const core::Geometry& geometry) {
    constexpr auto vec_size = 9 * sizeof(float);
    const auto size = sizeof(core::GlobalHeader) + geometry.num_devices() * vec_size;
    std::vector<uint8_t> buf;
    buf.resize(size);

    auto* const uh = reinterpret_cast<core::GlobalHeader*>(&buf[0]);
    uh->msg_id = core::MSG_EMU_GEOMETRY_SET;
    uh->fpga_ctrl_flags = 0x00;
    uh->cpu_ctrl_flags = 0x00;
    uh->mod_size = 0x00;

    auto* const cursor = reinterpret_cast<float*>(&buf[sizeof(core::GlobalHeader)]);
    for (const auto& device : geometry) {
      const auto i = device.id();
      auto origin = device.begin()->cast<float>();
      auto right = device.x_direction().cast<float>();
      auto up = device.y_direction().cast<float>();
      cursor[9 * i] = origin.x();
      cursor[9 * i + 1] = origin.y();
      cursor[9 * i + 2] = origin.z();
      cursor[9 * i + 3] = right.x();
      cursor[9 * i + 4] = right.y();
      cursor[9 * i + 5] = right.z();
      cursor[9 * i + 6] = up.x();
      cursor[9 * i + 7] = up.y();
      cursor[9 * i + 8] = up.z();
    }

    return buf;
  }
};

core::LinkPtr Emulator::create(const uint16_t port, const core::Geometry& geometry) {
  core::LinkPtr link = std::make_unique<EmulatorImpl>(port, geometry);
  return link;
}

EmulatorImpl::EmulatorImpl(const uint16_t port, const core::Geometry& geometry)
    : _is_open(false), _port(port), _geometry_buf(init_geometry_buf(geometry)) {}

void EmulatorImpl::send(const uint8_t* buf, const size_t size) {
  const auto* header = reinterpret_cast<const core::GlobalHeader*>(buf);
  _last_msg_id = header->msg_id;
  if (sendto(_socket, reinterpret_cast<const char*>(buf), static_cast<int>(size), 0, reinterpret_cast<sockaddr*>(&_addr), sizeof _addr) == -1)
    throw core::exception::LinkError("failed to send data");
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
    throw core::exception::LinkError("cannot connect to emulator");

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

void EmulatorImpl::receive(uint8_t* rx, size_t buffer_len) {
  for (size_t i = 0; i < buffer_len; i += 2) rx[i + 1] = this->_last_msg_id;

  const auto set = [rx, buffer_len](const uint8_t value) {
    for (size_t i = 0; i < buffer_len; i += 2) rx[i] = value;
  };

  switch (this->_last_msg_id) {
    case core::MSG_CLEAR:
      break;
    case core::MSG_RD_CPU_V_LSB:
    case core::MSG_RD_CPU_V_MSB:
    case core::MSG_RD_FPGA_V_LSB:
    case core::MSG_RD_FPGA_V_MSB:
      set(0xFF);
      break;
    default:
      break;
  }
}

bool EmulatorImpl::is_open() { return _is_open; }

}  // namespace autd::link
