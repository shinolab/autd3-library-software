// File: c_api.cpp
// Project: emulator_link
// Created Date: 07/07/2021
// Author: Shun Suzuki
// -----
// Last Modified: 10/12/2021
// Modified By: Shun Suzuki (suzuki@hapis.k.u-tokyo.ac.jp)
// -----
// Copyright (c) 2021 Hapis Lab. All rights reserved.
//

#include <autd3/link/emulator.hpp>

#include "../base/wrapper.hpp"
#include "../base/wrapper_link.hpp"
#include "./emulator_link.h"

void AUTDLinkEmulator(void** out, const uint16_t port, const void* cnt) {
  const auto* const p_cnt = static_cast<const autd::Controller*>(cnt);
  auto* link = LinkCreate(autd::link::Emulator::create(port, p_cnt->geometry()));
  *out = link;
}
