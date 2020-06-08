// File: core.hpp
// Project: include
// Created Date: 11/04/2018
// Author: Shun Suzuki
// -----
// Last Modified: 09/06/2020
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

class Link;

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

namespace modulation {
class Modulation;
class SineModulation;
class SawModulation;
class RawPCMModulation;
class WavModulation;
}  // namespace modulation

class FirmwareInfo;

using EtherCATAdapter = std::pair<std::string, std::string>;
#if DLL_FOR_CAPI
using GainPtr = gain::Gain *;
using ModulationPtr = modulation::Modulation *;
using LinkPtr = Link *;
using EtherCATAdapters = EtherCATAdapter *;
using ControllerPtr = Controller *;
using FirmwareInfoList = FirmwareInfo *;

template <class T>
static T *CreateHelper() {
  struct impl : T {
    impl() : T() {}
  };
  return new impl;
}
template <class T>
static void DeleteHelper(T **ptr) {
  delete *ptr;
  *ptr = nullptr;
}
#else
using GainPtr = std::shared_ptr<gain::Gain>;
using LinkPtr = std::shared_ptr<Link>;
using ModulationPtr = std::shared_ptr<modulation::Modulation>;
using EtherCATAdapters = std::vector<EtherCATAdapter>;
using ControllerPtr = std::shared_ptr<Controller>;
using FirmwareInfoList = std::vector<FirmwareInfo>;

template <class T>
static std::shared_ptr<T> CreateHelper() {
  struct impl : T {
    impl() : T() {}
  };
  auto p = std::make_shared<impl>();
  return std::move(p);
}
template <class T>
static void DeleteHelper(std::shared_ptr<T>* ptr) {
  *ptr = nullptr;
}
#endif
}  // namespace autd
