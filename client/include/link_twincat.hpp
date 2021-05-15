// File: link_twincat.hpp
// Project: include
// Created Date: 10/05/2021
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
   * @brief Create TwinCATLink.
   */
  static core::LinkPtr Create();

  TwinCATLink() : _port(0) {}
  ~TwinCATLink() override = default;
  TwinCATLink(const TwinCATLink& v) noexcept = delete;
  TwinCATLink& operator=(const TwinCATLink& obj) = delete;
  TwinCATLink(TwinCATLink&& obj) = delete;
  TwinCATLink& operator=(TwinCATLink&& obj) = delete;

  Result<bool, std::string> Open() override;
  Result<bool, std::string> Close() override;
  Result<bool, std::string> Send(size_t size, const uint8_t* buf) override;
  Result<bool, std::string> Read(uint8_t* rx, size_t buffer_len) override;
  bool is_open() override;

 private:
  long _port;  // NOLINT
#ifdef _WIN32
  AmsNetId _net_id{};
  HMODULE _lib = nullptr;
#endif
};
}  // namespace autd::link
