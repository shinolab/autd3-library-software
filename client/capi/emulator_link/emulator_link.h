// File: emulator_link.h
// Project: emulator_link
// Created Date: 07/07/2021
// Author: Shun Suzuki
// -----
// Last Modified: 10/12/2021
// Modified By: Shun Suzuki (suzuki@hapis.k.u-tokyo.ac.jp)
// -----
// Copyright (c) 2021 Hapis Lab. All rights reserved.
//

#pragma once

#include <cstdint>

#include "../base/header.h"

extern "C" {
EXPORT_AUTD void AUTDLinkEmulator(void** out, uint16_t port, const void* cnt);
}
