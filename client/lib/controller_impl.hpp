// File: controller_impl.hpp
// Project: lib
// Created Date: 11/10/2020
// Author: Shun Suzuki
// -----
// Last Modified: 11/10/2020
// Modified By: Shun Suzuki (suzuki@hapis.k.u-tokyo.ac.jp)
// -----
// Copyright (c) 2020 Hapis Lab. All rights reserved.
//

#include "controller.hpp"

namespace autd {
class AUTDControllerV_0_1 : public Controller {
 public:
  AUTDControllerV_0_1();
  ~AUTDControllerV_0_1() override;

  bool is_open() final;
  GeometryPtr geometry() noexcept final;
  bool silent_mode() noexcept final;
  size_t remaining_in_buffer() final;

  void OpenWith(LinkPtr link) final;
  void SetSilentMode(bool silent) noexcept final;
  bool Calibrate() final;
  bool Clear() final;
  void Close() final;

  void Stop() final;
  void AppendGain(const GainPtr gain) final;
  void AppendGainSync(const GainPtr gain, bool wait_for_send = false) final;
  void AppendModulation(const ModulationPtr mod) final;
  void AppendModulationSync(const ModulationPtr mod) final;
  void AppendSTMGain(GainPtr gain) final;
  void AppendSTMGain(const std::vector<GainPtr> &gain_list) final;
  void StartSTModulation(double freq) final;
  void StopSTModulation() final;
  void FinishSTModulation() final;
  void AppendSequence(SequencePtr seq) final;
  void Flush() final;
  FirmwareInfoList firmware_info_list() final;

  void LateralModulationAT(Vector3 point, Vector3 dir, double lm_amp = 2.5, double lm_freq = 100) final;

 private:
  GeometryPtr _geometry;
  LinkPtr _link;
  std::queue<GainPtr> _build_q;
  std::queue<GainPtr> _send_gain_q;
  std::queue<ModulationPtr> _send_mod_q;

  std::vector<GainPtr> _stm_gains;
  std::vector<uint8_t *> _stm_bodies;
  std::vector<size_t> _stm_body_sizes;
  std::unique_ptr<Timer> _p_stm_timer;

  std::thread _build_thr;
  std::thread _send_thr;
  std::condition_variable _build_cond;
  std::condition_variable _send_cond;
  std::mutex _build_mtx;
  std::mutex _send_mtx;

  std::vector<uint8_t> _rx_data;

  bool _silent_mode = true;

  void InitPipeline();
  std::unique_ptr<uint8_t[]> MakeBody(GainPtr gain, ModulationPtr mod, size_t *const size, uint8_t *const send_msg_id);
  bool WaitMsgProcessed(uint8_t msg_id, size_t max_trial = 200, uint8_t mask = 0xFF);

  std::unique_ptr<uint8_t[]> MakeSeqBody(SequencePtr seq, size_t *const size, uint8_t *const send_msg_id);
  void CalibrateSeq();
  std::unique_ptr<uint8_t[]> MakeCalibBody(std::vector<uint16_t> diffs, size_t *const size);
};
