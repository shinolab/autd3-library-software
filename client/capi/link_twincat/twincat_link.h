﻿// File: autd3_c_api_twincat_link.h
// Project: twincat_link
// Created Date: 20/02/2021
// Author: Shun Suzuki
// -----
// Last Modified: 20/02/2021
// Modified By: Shun Suzuki (suzuki@hapis.k.u-tokyo.ac.jp)
// -----
// Copyright (c) 2021 Hapis Lab. All rights reserved.
//

#pragma once

#if WIN32
#define EXPORT_AUTD __declspec(dllexport)
#else
#define EXPORT_AUTD __attribute__((visibility("default")))
#endif

#define VOID_PTR void*

extern "C" {
EXPORT_AUTD void AUTDTwinCATLink(VOID_PTR* out, const char* ipv4_addr, const char* ams_net_id);
EXPORT_AUTD void AUTDLocalTwinCATLink(VOID_PTR* out);
}
