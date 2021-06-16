// File: autd3.hpp
// Project: include
// Created Date: 14/04/2021
// Author: Shun Suzuki
// -----
// Last Modified: 16/06/2021
// Modified By: Shun Suzuki (suzuki@hapis.k.u-tokyo.ac.jp)
// -----
// Copyright (c) 2021 Hapis Lab. All rights reserved.
//

#pragma once

#include "controller.hpp"
#include "core/gain.hpp"
#include "core/hardware_defined.hpp"
#include "core/modulation.hpp"
#include "core/sequence.hpp"

namespace autd {
using Matrix4X4 = core::Matrix4X4;
using Quaternion = core::Quaternion;
using Vector3 = core::Vector3;
using LinkPtr = core::LinkPtr;
using GeometryPtr = core::GeometryPtr;
using ModulationPtr = core::ModulationPtr;
using GainPtr = core::GainPtr;
using SequencePtr = core::SequencePtr;
using DataArray = core::DataArray;

constexpr auto NUM_TRANS_IN_UNIT = core::NUM_TRANS_IN_UNIT;
constexpr auto NUM_TRANS_X = core::NUM_TRANS_X;
constexpr auto NUM_TRANS_Y = core::NUM_TRANS_Y;
constexpr auto TRANS_SPACING_MM = core::TRANS_SPACING_MM;
constexpr auto AUTD_WIDTH = core::AUTD_WIDTH;
constexpr auto AUTD_HEIGHT = core::AUTD_HEIGHT;
constexpr auto ULTRASOUND_FREQUENCY = core::ULTRASOUND_FREQUENCY;
}  // namespace autd
