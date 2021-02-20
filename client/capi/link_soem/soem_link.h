// File: autd3_c_api_soem_link.h
// Project: soem_link
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
EXPORT_AUTD int32_t AUTDGetAdapterPointer(VOID_PTR* out);
EXPORT_AUTD void AUTDGetAdapter(VOID_PTR p_adapter, int32_t index, char* desc, char* name);
EXPORT_AUTD void AUTDFreeAdapterPointer(VOID_PTR p_adapter);
EXPORT_AUTD void AUTDSOEMLink(VOID_PTR* out, const char* ifname, int32_t device_num);
}
