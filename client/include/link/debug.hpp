// File: debug_link.hpp
// Project: include
// Created Date: 22/12/2020
// Author: Shun Suzuki
// -----
// Last Modified: 01/04/2021
// Modified By: Shun Suzuki (suzuki@hapis.k.u-tokyo.ac.jp)
// -----
// Copyright (c) 2020 Hapis Lab. All rights reserved.
//

#pragma once

#include <memory>
#include <ostream>
#include <string>

#include "link.hpp"

namespace autd::link {
/**
 * @brief Link for debug
 */
class DebugLink final : public Link {
 public:
  static LinkPtr Create(std::ostream& out);

  explicit DebugLink(std::ostream& out);
  ~DebugLink() override = default;
  DebugLink(const DebugLink& v) noexcept = delete;
  DebugLink& operator=(const DebugLink& obj) = delete;
  DebugLink(DebugLink&& obj) = delete;
  DebugLink& operator=(DebugLink&& obj) = delete;

  void Open() override;
  void Close() override;
  std::optional<std::string> Send(size_t size, std::unique_ptr<uint8_t[]> buf) override;
  std::optional<std::string> Read(uint8_t* rx, uint32_t buffer_len) override;
  bool is_open() override;

 private:
  std::ostream& _out;
  bool _is_open = false;
  uint8_t _last_msg_id = 0;
};
}  // namespace autd::link
