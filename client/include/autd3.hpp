// File: autd3.hpp
// Project: include
// Created Date: 14/04/2021
// Author: Shun Suzuki
// -----
// Last Modified: 21/11/2021
// Modified By: Shun Suzuki (suzuki@hapis.k.u-tokyo.ac.jp)
// -----
// Copyright (c) 2021 Hapis Lab. All rights reserved.
//

#pragma once

#include "autd3/controller.hpp"
#include "autd3/core/gain.hpp"
#include "autd3/core/hardware_defined.hpp"
#include "autd3/core/modulation.hpp"
#include "autd3/core/sequence.hpp"
#include "autd3/gain/primitive.hpp"
#include "autd3/modulation/primitive.hpp"
#include "autd3/sequence/primitive.hpp"

namespace autd {
using core::DataArray;
using core::Geometry;
using core::LinkPtr;
using core::Matrix4X4;
using core::Quaternion;
using core::Vector3;
using gain::GainPtr;
using modulation::ModulationPtr;
using sequence::GAIN_MODE;
using sequence::GainSequencePtr;
using sequence::PointSequencePtr;

using core::DEVICE_HEIGHT;
using core::DEVICE_WIDTH;
using core::NUM_TRANS_IN_UNIT;
using core::NUM_TRANS_X;
using core::NUM_TRANS_Y;
using core::TRANS_SPACING_MM;
using core::ULTRASOUND_FREQUENCY;
}  // namespace autd
