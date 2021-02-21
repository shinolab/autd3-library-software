// File: c_api.cpp
// Project: capi
// Created Date: 02/07/2018
// Author: Shun Suzuki
// -----
// Last Modified: 21/02/2021
// Modified By: Shun Suzuki (suzuki@hapis.k.u-tokyo.ac.jp)
// -----
// Copyright (c) 2018-2020 Hapis Lab. All rights reserved.
//

#include "../base/wrapper_sequence.hpp"
#include "./sequence_primitive.h"
#include "sequence/primitive.hpp"

void AUTDCircumSequence(VOID_PTR* out, const autd::Float x, const autd::Float y, const autd::Float z, const autd::Float nx, const autd::Float ny,
                        const autd::Float nz, const autd::Float radius, const uint64_t n) {
  auto* s = SequencePtrCreate(autd::sequence::CircumSeq::Create(autd::Vector3(x, y, z), autd::Vector3(nx, ny, nz), radius, static_cast<size_t>(n)));
  *out = s;
}
