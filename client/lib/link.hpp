// File: link.hpp
// Project: lib
// Created Date: 01/06/2016
// Author: Seki Inoue
// -----
// Last Modified: 29/03/2020
// Modified By: Shun Suzuki (suzuki@hapis.k.u-tokyo.ac.jp)
// -----
// Copyright (c) 2016-2020 Hapis Lab. All rights reserved.
//

#pragma once

#include <stdio.h>

#include <memory>
#include <string>
#include <vector>

#include "gain.hpp"
#include "modulation.hpp"

namespace autd {
namespace internal {
class Link {
 public:
  virtual void Open(std::string location) = 0;
  virtual void Close() = 0;
  virtual void Send(size_t size, std::unique_ptr<uint8_t[]> buf) = 0;
  virtual std::vector<uint8_t> Read() = 0;
  virtual bool is_open() = 0;
};
}  // namespace internal

static inline std::vector<std::string> split(const std::string &s, char delim) {
  std::vector<std::string> tokens;
  std::string token;
  for (char ch : s) {
    if (ch == delim) {
      if (!token.empty()) tokens.push_back(token);
      token.clear();
    } else {
      token += ch;
    }
  }
  if (!token.empty()) tokens.push_back(token);
  return tokens;
}

}  // namespace autd
