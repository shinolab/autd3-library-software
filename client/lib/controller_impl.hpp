// File: controller_impl.hpp
// Project: lib
// Created Date: 11/10/2020
// Author: Shun Suzuki
// -----
// Last Modified: 03/04/2021
// Modified By: Shun Suzuki (suzuki@hapis.k.u-tokyo.ac.jp)
// -----
// Copyright (c) 2020 Hapis Lab. All rights reserved.
//

#pragma once

#include <memory>
#include <string>
#include <vector>

#include "autd_logic.hpp"
#include "configuration.hpp"
#include "controller.hpp"

namespace autd::internal {

class AUTDControllerSync;
class AUTDControllerAsync;
class AUTDControllerStm;

class AUTDController final : public Controller {
 public:
  AUTDController();
  ~AUTDController() override;
  AUTDController(const AUTDController& v) noexcept = delete;
  AUTDController& operator=(const AUTDController& obj) = delete;
  AUTDController(AUTDController&& obj) = delete;
  AUTDController& operator=(AUTDController&& obj) = delete;

  bool is_open() override;
  GeometryPtr geometry() noexcept override;
  bool silent_mode() noexcept override;
  size_t remaining_in_buffer() override;

  void SetSilentMode(bool silent) noexcept override;

  Result<bool, std::string> OpenWith(LinkPtr link) override;
  Result<bool, std::string> Calibrate(Configuration config) override;
  Result<bool, std::string> Synchronize(Configuration config) override;
  Result<bool, std::string> Clear() override;
  Result<bool, std::string> Close() override;
  void Flush() override;

  Result<bool, std::string> Stop() override;
  void AppendGain(GainPtr gain) override;
  Result<bool, std::string> AppendGainSync(GainPtr gain, bool wait_for_send = false) override;
  void AppendModulation(ModulationPtr mod) override;
  Result<bool, std::string> AppendModulationSync(ModulationPtr mod) override;
  void AppendSTMGain(GainPtr gain) override;
  void AppendSTMGain(const std::vector<GainPtr>& gain_list) override;
  Result<bool, std::string> StartSTModulation(Float freq) override;
  void StopSTModulation() override;
  void FinishSTModulation() override;
  Result<bool, std::string> AppendSequence(SequencePtr seq) override;
  Result<std::vector<FirmwareInfo>, std::string> firmware_info_list() override;

 private:
  std::unique_ptr<AUTDControllerSync> _sync_cnt;
  std::unique_ptr<AUTDControllerAsync> _async_cnt;
  std::unique_ptr<AUTDControllerStm> _stm_cnt;
  std::shared_ptr<AUTDLogic> _autd_logic;
};
}  // namespace autd::internal
