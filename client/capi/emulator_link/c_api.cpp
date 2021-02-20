// File: c_api.cpp
// Project: capi
// Created Date: 02/07/2018
// Author: Shun Suzuki
// -----
// Last Modified: 20/02/2021
// Modified By: Shun Suzuki (suzuki@hapis.k.u-tokyo.ac.jp)
// -----
// Copyright (c) 2018-2020 Hapis Lab. All rights reserved.
//

#include "./autd3_c_api_emulator_link.h"
#include "emulator_link.hpp"
#include "../base/wrapper.hpp"
#include "../base/wrapper_link.hpp"

void AUTDEmulatorLink(VOID_PTR* out, const char* addr, const uint16_t port, VOID_PTR const handle) {
  auto* cnt = static_cast<ControllerWrapper*>(handle);
  auto* link = LinkCreate(autd::link::EmulatorLink::Create(std::string(addr), port, cnt->ptr->geometry()));
  *out = link;
}
