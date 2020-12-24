// File: debug_link.hpp
// Project: include
// Created Date: 22/12/2020
// Author: Shun Suzuki
// -----
// Last Modified: 24/12/2020
// Modified By: Shun Suzuki (suzuki@hapis.k.u-tokyo.ac.jp)
// -----
// Copyright (c) 2020 Hapis Lab. All rights reserved.
//

#pragma once

#include <memory>
#include <ostream>
#include <string>
#include <vector>

#include "link.hpp"

namespace autd::link {
/**
 * @brief Link for debug
 */
class DebugLink : public Link {
 public:
  static LinkPtr Create(std::ostream& out);

  explicit DebugLink(std::ostream& out);
  ~DebugLink() override{};

  void Open() final;
  void Close() final;
  void Send(size_t size, std::unique_ptr<uint8_t[]> buf) final;
  std::vector<uint8_t> Read(uint32_t buffer_len) final;
  bool is_open() final;

 private:
  std::ostream& _out;
  bool _is_open = false;
  size_t _dev_num = 0;
  uint8_t _last_msg_id = 0;
};
}  // namespace autd::link
