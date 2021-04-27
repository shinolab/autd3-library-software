// File: c_api.cpp
// Project: capi
// Created Date: 02/07/2018
// Author: Shun Suzuki
// -----
// Last Modified: 05/04/2021
// Modified By: Shun Suzuki (suzuki@hapis.k.u-tokyo.ac.jp)
// -----
// Copyright (c) 2018-2020 Hapis Lab. All rights reserved.
//

#include "../base/wrapper.hpp"
#include "../base/wrapper_link.hpp"
#include "./emulator_link.h"
#include "link/emulator.hpp"

void AUTDEmulatorLink(void** out, const char* addr, const uint16_t port, void* const handle) {
  auto* cnt = static_cast<ControllerWrapper*>(handle);
  auto* link = LinkCreate(autd::link::EmulatorLink::Create(std::string(addr), port, cnt->ptr->geometry()));
  *out = link;
}
