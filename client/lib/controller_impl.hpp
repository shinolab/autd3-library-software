// File: controller_impl.hpp
// Project: lib
// Created Date: 11/10/2020
// Author: Shun Suzuki
// -----
// Last Modified: 22/12/2020
// Modified By: Shun Suzuki (suzuki@hapis.k.u-tokyo.ac.jp)
// -----
// Copyright (c) 2020 Hapis Lab. All rights reserved.
//

#pragma once

#include <atomic>
#include <memory>
#include <queue>
#include <vector>

#include "autd_logic.hpp"
#include "configuration.hpp"
#include "controller.hpp"
#include "ec_config.hpp"
#include "firmware_version.hpp"

namespace autd::_internal {

class AUTDControllerSync;
class AUTDControllerAsync;
class AUTDControllerSTM;

class AUTDController : public Controller {
 public:
  AUTDController();
  virtual ~AUTDController();

  bool is_open() final;
  GeometryPtr geometry() noexcept final;
  bool silent_mode() noexcept final;
  size_t remaining_in_buffer() final;
  void SetSilentMode(bool silent) noexcept final;

  void OpenWith(LinkPtr link) final;
  bool Calibrate(Configuration config);
  bool Clear();
  void Close();
  void Flush();

  void Stop();
  void AppendGain(const GainPtr gain);
  void AppendGainSync(const GainPtr gain, bool wait_for_send = false);
  void AppendModulation(const ModulationPtr mod);
  void AppendModulationSync(const ModulationPtr mod);
  void AppendSTMGain(GainPtr gain);
  void AppendSTMGain(const std::vector<GainPtr> &gain_list);
  void StartSTModulation(double freq);
  void StopSTModulation();
  void FinishSTModulation();
  void AppendSequence(SequencePtr seq);
  FirmwareInfoList firmware_info_list();

  void LateralModulationAT(Vector3 point, Vector3 dir, double lm_amp = 2.5, double lm_freq = 100);

 private:
  std::unique_ptr<_internal::AUTDControllerSync> _sync_cnt;
  std::unique_ptr<_internal::AUTDControllerAsync> _async_cnt;
  std::unique_ptr<_internal::AUTDControllerSTM> _stm_cnt;
  std::shared_ptr<_internal::AUTDLogic> _autd_logic;
};
}  // namespace autd::_internal
