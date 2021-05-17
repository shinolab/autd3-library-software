// File: twincat_link.hpp
// Project: include
// Created Date: 10/05/2021
// Author: Shun Suzuki
// -----
// Last Modified: 17/05/2021
// Modified By: Shun Suzuki (suzuki@hapis.k.u-tokyo.ac.jp)
// -----
// Copyright (c) 2021 Hapis Lab. All rights reserved.
//

#pragma once

#include <memory>

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
class TwinCATLink final : public core::Link {
 public:
  /**
   * @brief Create TwinCATLink
   */
  static core::LinkPtr Create();

  TwinCATLink() : _port(0) {}
  ~TwinCATLink() override = default;
  TwinCATLink(const TwinCATLink& v) noexcept = delete;
  TwinCATLink& operator=(const TwinCATLink& obj) = delete;
  TwinCATLink(TwinCATLink&& obj) = delete;
  TwinCATLink& operator=(TwinCATLink&& obj) = delete;

  Error Open() override;
  Error Close() override;
  Error Send(size_t size, const uint8_t* buf) override;
  Error Read(uint8_t* rx, size_t buffer_len) override;
  bool is_open() override;

 private:
  long _port;  // NOLINT
#ifdef _WIN32
  AmsNetId _net_id{};
  HMODULE _lib = nullptr;
#endif
};
}  // namespace autd::link
