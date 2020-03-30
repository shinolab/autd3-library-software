// File: controller.hpp
// Project: include
// Created Date: 11/04/2018
// Author: Shun Suzuki
// -----
// Last Modified: 30/03/2020
// Modified By: Shun Suzuki (suzuki@hapis.k.u-tokyo.ac.jp)
// -----
// Copyright (c) 2018-2020 Hapis Lab. All rights reserved.
//

#pragma once

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "gain.hpp"
#include "geometry.hpp"
#include "modulation.hpp"
#include "vector3.hpp"

namespace autd {

class Controller;

using EtherCATAdapter = std::pair<std::string, std::string>;
#if DLL_FOR_CAPI
using EtherCATAdapters = EtherCATAdapter *;
using ControllerPtr = Controller *;
#else
using EtherCATAdapters = std::vector<EtherCATAdapter>;
using ControllerPtr = std::shared_ptr<Controller>;
#endif

class Controller {
 public:
  static ControllerPtr Create();

  virtual bool is_open() = 0;
  virtual GeometryPtr geometry() noexcept = 0;
  virtual bool silent_mode() noexcept = 0;
  virtual size_t remainingInBuffer() = 0;

  static EtherCATAdapters EnumerateAdapters(int *const size);

  /**
   * @brief Open device by link type and location.
   * The scheme of location is as follows:
   * ETHERCAT - <ams net id> or <ipv4 addr>:<ams net id> (ex. 192.168.1.2:192.168.1.3.1.1 ).
   *  The ipv4 addr will be extracted from leading 4 octets of ams net id if not specified.
   * ETHERNET - ipv4 addr
   * USB      - ignored
   * SERIAL   - file discriptor
   */
  virtual void Open(LinkType type, std::string location = "") = 0;
  virtual void SetSilentMode(bool silent) noexcept = 0;
  virtual bool CalibrateModulation() = 0;
  virtual void Close() = 0;

  virtual void Stop() = 0;
  virtual void AppendGain(GainPtr gain) = 0;
  virtual void AppendGainSync(GainPtr gain) = 0;
  virtual void AppendModulation(ModulationPtr modulation) = 0;
  virtual void AppendModulationSync(ModulationPtr modulation) = 0;
  virtual void AppendSTMGain(GainPtr gain) = 0;
  virtual void AppendSTMGain(const std::vector<GainPtr> &gain_list) = 0;
  virtual void StartSTModulation(double freq) = 0;
  virtual void StopSTModulation() = 0;
  virtual void FinishSTModulation() = 0;
  virtual void Flush() = 0;

  virtual void LateralModulationAT(Vector3 point, Vector3 dir = Vector3::unit_y(), double lm_amp = 2.5, double lm_freq = 100) = 0;
};
}  // namespace autd
