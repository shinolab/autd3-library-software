// File: autd3_c_api_soem_link.h
// Project: soem_link
// Created Date: 20/02/2021
// Author: Shun Suzuki
// -----
// Last Modified: 18/05/2021
// Modified By: Shun Suzuki (suzuki@hapis.k.u-tokyo.ac.jp)
// -----
// Copyright (c) 2021 Hapis Lab. All rights reserved.
//

#pragma once

#include <cstdint>

#include "../base/header.h"

extern "C" {
EXPORT_AUTD int32_t AUTDGetAdapterPointer(void** out);
EXPORT_AUTD void AUTDGetAdapter(void* p_adapter, int32_t index, char* desc, char* name);
EXPORT_AUTD void AUTDFreeAdapterPointer(void* p_adapter);
EXPORT_AUTD void AUTDSOEMLink(void** out, const char* ifname, int32_t device_num);
}
