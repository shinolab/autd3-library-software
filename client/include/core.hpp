// File: core.hpp
// Project: include
// Created Date: 11/04/2018
// Author: Shun Suzuki
// -----
// Last Modified: 30/04/2020
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
namespace internal {
class Link;
}

namespace _utils {
class Vector3;
class Quaternion;
}  // namespace _utils
using _utils::Quaternion;
using _utils::Vector3;

enum class LinkType : int { ETHERCAT, TwinCAT, SOEM, EMULATOR };

class Controller;
class AUTDController;
class Geometry;
class Timer;

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
}  // namespace gain
// For Backward compatibility
using NullGain = gain::Gain;
using HoloGain = gain::HoloGainSdp;
using PlaneWaveGain = gain::PlaneWaveGain;
using FocalPointGain = gain::FocalPointGain;
using BesselBeamGain = gain::BesselBeamGain;
using CustomGain = gain::CustomGain;
using GroupedGain = gain::GroupedGain;
using HoloGainSdp = gain::HoloGainSdp;
using MatlabGain = gain::MatlabGain;
using TransducerTestGain = gain::TransducerTestGain;

namespace modulation {
class Modulation;
class SineModulation;
class SawModulation;
class RawPCMModulation;
class WavModulation;
}  // namespace modulation
// For Backward compatibility
using Modulation = modulation::Modulation;
using SineModulation = modulation::SineModulation;
using SawModulation = modulation::SawModulation;
using RawPCMModulation = modulation::RawPCMModulation;
using WavModulation = modulation::WavModulation;

#if DLL_FOR_CAPI
using GainPtr = gain::Gain*;
using ModulationPtr = modulation::Modulation*;

template <class T>
static T* CreateHelper() {
  struct impl : T {
    impl() : T() {}
  };
  return new impl;
}
template <class T>
static void DeleteHelper(T** ptr) {
  delete *ptr;
  *ptr = nullptr;
}
#else
using GainPtr = std::shared_ptr<gain::Gain>;
using ModulationPtr = std::shared_ptr<modulation::Modulation>;

template <class T>
static std::shared_ptr<T> CreateHelper() {
  struct impl : T {
    impl() : T() {}
  };
  auto p = std::make_shared<impl>();
  return std::move(p);
}
template <class T>
static void DeleteHelper(std::shared_ptr<T>* ptr) {}
#endif
}  // namespace autd
