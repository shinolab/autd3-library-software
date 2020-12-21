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
  bool Calibrate(Configuration config) = 0;
  bool Clear() = 0;
  void Close() = 0;

  void Stop() = 0;
  void AppendGain(const GainPtr gain) = 0;
  void AppendGainSync(const GainPtr gain, bool wait_for_send = false) = 0;
  void AppendModulation(const ModulationPtr mod) = 0;
  void AppendModulationSync(const ModulationPtr mod) = 0;
  void AppendSTMGain(GainPtr gain) = 0;
  void AppendSTMGain(const std::vector<GainPtr> &gain_list) = 0;
  void StartSTModulation(double freq) = 0;
  void StopSTModulation() = 0;
  void FinishSTModulation() = 0;
  void AppendSequence(SequencePtr seq) = 0;
  void Flush() = 0;
  FirmwareInfoList firmware_info_list() = 0;

  void LateralModulationAT(Vector3 point, Vector3 dir, double lm_amp = 2.5, double lm_freq = 100) = 0;

 private:
  static uint8_t get_id() {
    static std::atomic<uint8_t> id{OP_MODE_MSG_ID_MIN - 1};

    id.fetch_add(0x01);
    uint8_t expected = OP_MODE_MSG_ID_MAX + 1;
    id.compare_exchange_weak(expected, OP_MODE_MSG_ID_MIN);

    return id.load();
  }

  void CloseLink();
  void SendData(size_t size, std::unique_ptr<uint8_t[]> buf);
  std::vector<uint8_t> ReadData(uint32_t buffer_len);
  size_t &mod_sent(ModulationPtr mod);
  size_t &seq_sent(SequencePtr seq);
  uint16_t seq_div(SequencePtr seq);
  const uint16_t *gain_data_addr(GainPtr gain, int device_id);

  void InitPipeline();
  std::unique_ptr<uint8_t[]> MakeBody(GainPtr gain, ModulationPtr mod, size_t *const size, uint8_t *const send_msg_id);
  bool WaitMsgProcessed(uint8_t msg_id, size_t max_trial = 200, uint8_t mask = 0xFF);
  std::unique_ptr<uint8_t[]> MakeSeqBody(SequencePtr seq, size_t *const size, uint8_t *const send_msg_id);
  void CalibrateSeq();
  std::unique_ptr<uint8_t[]> MakeCalibBody(std::vector<uint16_t> diffs, size_t *const size);

  static FirmwareInfo FirmwareInfoCreate(uint16_t idx, uint16_t cpu_ver, uint16_t fpga_ver) { return FirmwareInfo{idx, cpu_ver, fpga_ver}; }

  GeometryPtr _geometry;
  LinkPtr _link;

  std::queue<GainPtr> _build_gain_q;
  std::queue<ModulationPtr> _build_mod_q;
  std::queue<GainPtr> _send_gain_q;
  std::queue<ModulationPtr> _send_mod_q;

  std::vector<GainPtr> _stm_gains;
  std::vector<uint8_t *> _stm_bodies;
  std::vector<size_t> _stm_body_sizes;
  std::unique_ptr<Timer> _p_stm_timer;

  std::thread _build_gain_thr;
  std::thread _build_mod_thr;
  std::thread _send_thr;
  std::condition_variable _build_gain_cond;
  std::condition_variable _build_mod_cond;
  std::condition_variable _send_cond;
  std::mutex _build_gain_mtx;
  std::mutex _build_mod_mtx;
  std::mutex _send_mtx;

  std::vector<uint8_t> _rx_data;
  bool _seq_mode;

  bool _silent_mode = true;
  Configuration _config = Configuration::GetDefaultConfiguration();
};
}  // namespace autd
