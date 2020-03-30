// File: soem_link.hpp
// Project: lib
// Created Date: 24/08/2019
// Author: Shun Suzuki
// -----
// Last Modified: 30/03/2020
// Modified By: Shun Suzuki (suzuki@hapis.k.u-tokyo.ac.jp)
// -----
// Copyright (c) 2019-2020 Hapis Lab. All rights reserved.
//

#pragma once

#include <memory>
#include <string>
#include <vector>

#include "autdsoem.hpp"
#include "link.hpp"

namespace autd {
namespace internal {
class SOEMLink : public Link {
 public:
  void Open(std::string location) final;
  void Close() final;
  void Send(size_t size, std::unique_ptr<uint8_t[]> buf) final;
  std::vector<uint8_t> Read(uint32_t buffer_len) final;
  bool is_open() final;
  bool CalibrateModulation() final;

 private:
  std::vector<uint8_t> WaitProcMsg(uint8_t id, uint8_t mask);
  std::unique_ptr<autdsoem::ISOEMController> _cnt;
  size_t _dev_num = 0;
  std::string _ifname;
  autdsoem::ECConfig _config{};
};
}  // namespace internal
}  // namespace autd
