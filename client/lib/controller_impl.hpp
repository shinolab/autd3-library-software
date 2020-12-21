// File: controller_impl.hpp
// Project: lib
// Created Date: 11/10/2020
// Author: Shun Suzuki
// -----
// Last Modified: 21/12/2020
// Modified By: Shun Suzuki (suzuki@hapis.k.u-tokyo.ac.jp)
// -----
// Copyright (c) 2020 Hapis Lab. All rights reserved.
//

#pragma once

#include <atomic>
#include <memory>
#include <queue>
#include <vector>

#include "configuration.hpp"
#include "controller.hpp"
#include "ec_config.hpp"
#include "firmware_version.hpp"

namespace autd {

namespace _internal {
class AUTDControllerSync;
class AUTDControllerAsync;
class AUTDLinkManager;
}  // namespace _internal

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
  void Flush();
  FirmwareInfoList firmware_info_list();

  void LateralModulationAT(Vector3 point, Vector3 dir, double lm_amp = 2.5, double lm_freq = 100);

 private:
  void CloseLink();

  void InitPipeline();

  static FirmwareInfo FirmwareInfoCreate(uint16_t idx, uint16_t cpu_ver, uint16_t fpga_ver) { return FirmwareInfo{idx, cpu_ver, fpga_ver}; }

  GeometryPtr _geometry;
  std::unique_ptr<AUTDLinkManager> _link_manager;

  // std::queue<GainPtr> _build_gain_q;
  // std::queue<ModulationPtr> _build_mod_q;
  // std::queue<GainPtr> _send_gain_q;
  // std::queue<ModulationPtr> _send_mod_q;

  std::vector<GainPtr> _stm_gains;
  std::vector<uint8_t *> _stm_bodies;
  std::vector<size_t> _stm_body_sizes;
  std::unique_ptr<Timer> _p_stm_timer;

  // std::thread _build_gain_thr;
  // std::thread _build_mod_thr;
  // std::thread _send_thr;
  // std::condition_variable _build_gain_cond;
  // std::condition_variable _build_mod_cond;
  // std::condition_variable _send_cond;
  // std::mutex _build_gain_mtx;
  // std::mutex _build_mod_mtx;
  // std::mutex _send_mtx;

  std::vector<uint8_t> _rx_data;
  bool _seq_mode;

  bool _silent_mode = true;
  Configuration _config = Configuration::GetDefaultConfiguration();
};

}  // namespace autd
