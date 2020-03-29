// File: soem_link.hpp
// Project: lib
// Created Date: 24/08/2019
// Author: Shun Suzuki
// -----
// Last Modified: 29/03/2020
// Modified By: Shun Suzuki (suzuki@hapis.k.u-tokyo.ac.jp)
// -----
// Copyright (c) 2019-2020 Hapis Lab. All rights reserved.
//

#pragma once

#include <memory>
#include <string>

#include "autdsoem.hpp"
#include "link.hpp"

namespace autd {
namespace internal {
class SOEMLink : public Link {
 public:
  void Open(std::string location) final;
  void Close() final;
  void Send(size_t size, std::unique_ptr<uint8_t[]> buf) final;
  virtual std::vector<uint8_t> Read() final;
  bool is_open() final;

 private:
  std::unique_ptr<autdsoem::ISOEMController> _cnt;
};
}  // namespace internal
}  // namespace autd
