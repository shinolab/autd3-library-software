// File: type_hints.hpp
// Project: core
// Created Date: 09/12/2021
// Author: Shun Suzuki
// -----
// Last Modified: 09/12/2021
// Modified By: Shun Suzuki (suzuki@hapis.k.u-tokyo.ac.jp)
// -----
// Copyright (c) 2021 Hapis Lab. All rights reserved.
//

#pragma once

#include "gain.hpp"
#include "modulation.hpp"
#include "sequence.hpp"

namespace autd::core {

template <typename T>
struct is_modulation : std::is_base_of<Modulation, T> {};
template <typename T>
struct is_gain : std::is_base_of<Gain, T> {};
template <typename T>
struct is_point_sequence : std::is_base_of<PointSequence, T> {};
template <typename T>
struct is_gain_sequence : std::is_base_of<GainSequence, T> {};

}  // namespace autd::core
