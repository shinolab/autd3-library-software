// File: modulation.hpp
// Project: include
// Created Date: 04/11/2018
// Author: Shun Suzuki
// -----
// Last Modified: 20/02/2020
// Modified By: Shun Suzuki (suzuki@hapis.k.u-tokyo.ac.jp)
// -----
// Copyright (c) 2018-2020 Hapis Lab. All rights reserved.
//

#pragma once

#include <memory>
#include <string>
#include <vector>

#include "core.hpp"

namespace autd {
class Modulation;

#if DLL_FOR_CAPI
using ModulationPtr = Modulation *;
#else
using ModulationPtr = std::shared_ptr<Modulation>;
#endif

class Modulation {
  friend class AUTDController;

 public:
  Modulation() noexcept;
  static ModulationPtr Create();
  static ModulationPtr Create(uint8_t amp);
  constexpr static int samplingFrequency();
  std::vector<uint8_t> buffer;

 private:
  size_t sent;
};

class SineModulation : public Modulation {
 public:
  static ModulationPtr Create(int freq, double amp = 1.0, double offset = 0.5);
};

class SawModulation : public Modulation {
 public:
  static ModulationPtr Create(int freq);
};

class RawPCMModulation : public Modulation {
 public:
  static ModulationPtr Create(std::string filename, double samplingFreq = 0.0);
};
}  // namespace autd
