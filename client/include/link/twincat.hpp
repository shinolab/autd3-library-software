// File: twincat.hpp
// Project: link
// Created Date: 01/05/2021
// Author: Shun Suzuki
// -----
// Last Modified: 11/05/2021
// Modified By: Shun Suzuki (suzuki@hapis.k.u-tokyo.ac.jp)
// -----
// Copyright (c) 2021 Hapis Lab. All rights reserved.
//

#pragma once

#include <memory>
#include <string>

#if _WINDOWS
#define NOMINMAX
#include <Windows.h>
#else
typedef void* HMODULE;
#endif

#include "core/link.hpp"

namespace autd::link {

struct AmsNetId {
  uint8_t b[6];
};

struct AmsAddr {
  AmsNetId net_id;
  uint16_t port;
};

/**
 * @brief TwinCATLink using local TwinCAT server
 */
class TwinCATLink : public core::Link {
 public:
  /**
   * @brief Create LocalTwinCATLink.
   */
  static core::LinkPtr Create();

  TwinCATLink() : _port(0), _net_id() {}
  ~TwinCATLink() override = default;
  TwinCATLink(const TwinCATLink& v) noexcept = delete;
  TwinCATLink& operator=(const TwinCATLink& obj) = delete;
  TwinCATLink(TwinCATLink&& obj) = delete;
  TwinCATLink& operator=(TwinCATLink&& obj) = delete;

  Result<bool, std::string> Open() override = 0;
  Result<bool, std::string> Close() override = 0;
  Result<bool, std::string> Send(size_t size, const uint8_t* buf) override = 0;
  Result<bool, std::string> Read(uint8_t* rx, uint32_t buffer_len) override = 0;
  bool is_open() override = 0;

 private:
  long _port;  // NOLINT
  AmsNetId _net_id;
#ifdef _WIN32
  HMODULE _lib = nullptr;
#endif
};
}  // namespace autd::link
