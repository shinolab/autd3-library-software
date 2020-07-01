// File: core.hpp
// Project: include
// Created Date: 11/04/2018
// Author: Shun Suzuki
// -----
// Last Modified: 01/07/2020
// Modified By: Shun Suzuki (suzuki@hapis.k.u-tokyo.ac.jp)
// -----
// Copyright (c) 2018-2020 Hapis Lab. All rights reserved.
//

#pragma once

#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace autd {

namespace _utils {
class Vector3;
class Quaternion;
}  // namespace _utils

using _utils::Quaternion;
using _utils::Vector3;

class Controller;
class AUTDController;
class Geometry;
class Timer;

namespace link {
class Link;
}  // namespace link

namespace gain {
class Gain;
class PlaneWaveGain;
class FocalPointGain;
class BesselBeamGain;
class CustomGain;
class GroupedGain;
class HoloGainSdp;
class MatlabGain;
class TransducerTestGain;
using NullGain = Gain;
using HoloGain = HoloGainSdp;
}  // namespace gain

namespace modulation {
class Modulation;
class SineModulation;
class SawModulation;
class RawPCMModulation;
class WavModulation;
}  // namespace modulation

namespace sequence {
class PointSequence;
class CircumSeq;
}  // namespace sequence

class FirmwareInfo;

using GainPtr = std::shared_ptr<gain::Gain>;
using LinkPtr = std::shared_ptr<link::Link>;
using ModulationPtr = std::shared_ptr<modulation::Modulation>;
using SequencePtr = std::shared_ptr<sequence::PointSequence>;
using ControllerPtr = std::shared_ptr<Controller>;
using FirmwareInfoList = std::vector<FirmwareInfo>;
}  // namespace autd
