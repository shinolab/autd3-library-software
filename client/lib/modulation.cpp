// File: modulation.cpp
// Project: lib
// Created Date: 11/06/2016
// Author: Seki Inoue
// -----
// Last Modified: 20/02/2021
// Modified By: Shun Suzuki (suzuki@hapis.k.u-tokyo.ac.jp)
// -----
// Copyright (c) 2016-2020 Hapis Lab. All rights reserved.
//

#include "modulation.hpp"

#include "configuration.hpp"

using autd::MOD_BUF_SIZE;
using autd::MOD_SAMPLING_FREQ;

namespace autd::modulation {
Modulation::Modulation() noexcept { this->_sent = 0; }

ModulationPtr Modulation::Create(const uint8_t amp) {
  auto mod = std::make_shared<Modulation>();
  mod->buffer.resize(1, amp);
  return mod;
}

void Modulation::Build(Configuration config) {}

size_t& Modulation::sent() { return _sent; }
}  // namespace autd::modulation
