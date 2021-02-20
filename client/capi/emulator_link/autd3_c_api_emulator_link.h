// File: autd3_c_api_emulator_link.h
// Project: emulator_link
// Created Date: 20/02/2021
// Author: Shun Suzuki
// -----
// Last Modified: 20/02/2021
// Modified By: Shun Suzuki (suzuki@hapis.k.u-tokyo.ac.jp)
// -----
// Copyright (c) 2021 Hapis Lab. All rights reserved.
//

#pragma once

#include <cstdint>

#if WIN32
#define EXPORT_AUTD __declspec(dllexport)
#else
#define EXPORT_AUTD __attribute__((visibility("default")))
#endif

#define VOID_PTR void*

extern "C" {
EXPORT_AUTD void AUTDEmulatorLink(VOID_PTR* out, const char* addr, uint16_t port, VOID_PTR handle);
}
