// File: header.h
// Project: base
// Created Date: 18/05/2021
// Author: Shun Suzuki
// -----
// Last Modified: 13/12/2021
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

#ifdef __cplusplus
using bool_t = bool;
#include <cstdint>
#else
#include <stdint.h>
#ifndef bool_t
#define bool_t BOOL
#endif
#endif
