// File: autd3.hpp
// Project: include
// Created Date: 14/04/2021
// Author: Shun Suzuki
// -----
// Last Modified: 11/05/2021
// Modified By: Shun Suzuki (suzuki@hapis.k.u-tokyo.ac.jp)
// -----
// Copyright (c) 2021 Hapis Lab. All rights reserved.
//

#pragma once

#include "controller.hpp"
#include "core/configuration.hpp"
#include "core/firmware_version.hpp"
#include "core/gain.hpp"
#include "core/hardware_defined.hpp"
#include "core/modulation.hpp"
#include "primitive_gain.hpp"
#include "primitive_modulation.hpp"

namespace autd {
using Matrix4X4 = autd::core::Matrix4X4;
using Quaternion = autd::core::Quaternion;
using Vector3 = autd::core::Vector3;
using LinkPtr = autd::core::LinkPtr;
using ModulationPtr = autd::core::ModulationPtr;
using GainPtr = autd::core::GainPtr;

constexpr auto NUM_TRANS_IN_UNIT = autd::core::NUM_TRANS_IN_UNIT;
constexpr auto NUM_TRANS_X = autd::core::NUM_TRANS_X;
constexpr auto NUM_TRANS_Y = autd::core::NUM_TRANS_Y;
constexpr auto TRANS_SPACING_MM = autd::core::TRANS_SPACING_MM;
constexpr auto AUTD_WIDTH = autd::core::AUTD_WIDTH;
constexpr auto AUTD_HEIGHT = autd::core::AUTD_HEIGHT;
constexpr auto ULTRASOUND_FREQUENCY = autd::core::ULTRASOUND_FREQUENCY;
}  // namespace autd
