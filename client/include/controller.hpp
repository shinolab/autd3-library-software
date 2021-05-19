// File: controller.hpp
// Project: include
// Created Date: 05/11/2020
// Author: Shun Suzuki
// -----
// Last Modified: 19/05/2021
// Modified By: Shun Suzuki (suzuki@hapis.k.u-tokyo.ac.jp)
// -----
// Copyright (c) 2021 Hapis Lab. All rights reserved.
//

#pragma once

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "core/configuration.hpp"
#include "core/firmware_version.hpp"
#include "core/gain.hpp"
#include "core/geometry.hpp"
#include "core/link.hpp"
#include "core/logic.hpp"
#include "core/modulation.hpp"
#include "core/osal_timer.hpp"
#include "core/sequence.hpp"

namespace autd {

/**
 * @brief AUTD Controller
 */
class Controller {
 public:
  class STMController;

  Controller() noexcept
      : _link(nullptr),
        _geometry(std::make_shared<core::Geometry>()),
        _silent_mode(true),
        _read_fpga_info(false),
        _seq_mode(false),
        _config(core::Configuration::GetDefaultConfiguration()),
        _tx_buf(nullptr),
        _rx_buf(nullptr),
        _stm(nullptr) {}
  ~Controller() = default;
  Controller(const Controller& v) noexcept = delete;
  Controller& operator=(const Controller& obj) = delete;
  Controller(Controller&& obj) = delete;
  Controller& operator=(Controller&& obj) = delete;

  /**
   * @brief Verify the device is properly connected
   */
  [[nodiscard]] bool is_open() const;

  /**
   * @brief Geometry of the devices
   */
  [[nodiscard]] core::GeometryPtr geometry() const noexcept;

  /**
   * @brief Silent mode
   */
  bool& silent_mode() noexcept;

  /**
   * @brief If true, the devices return FPGA info in all frames. The FPGA info can be read by fpga_info().
   */
  bool& reads_fpga_info() noexcept;

  /**
   * @brief FPGA info
   */
  Result<std::vector<uint8_t>, std::string> fpga_info();

  /**
   * @brief Update control flag
   */
  [[nodiscard]] Error set_output_delay(const std::vector<core::DataArray>& delay) const;

  /**
   * @brief Open device with a link.
   * @param[in] link Link
   * @return return Ok(whether succeeded to open), or Err(error msg) if some unrecoverable error occurred
   */
  [[nodiscard]] Error OpenWith(const core::LinkPtr& link);

  /**
   * @brief Synchronize all devices
   * @details Call this function only once after OpenWith(). It takes several seconds and blocks the thread in the meantime.
   * @param[in] config configuration
   * @return return Ok(whether succeeded to synchronize), or Err(error msg) if some unrecoverable error occurred
   */
  [[nodiscard]] Error Synchronize(core::Configuration config = core::Configuration::GetDefaultConfiguration());

  /**
   * @brief Clear all data in hardware
   * @return return Ok(whether succeeded to clear), or Err(error msg) if some unrecoverable error occurred
   */
  [[nodiscard]] Error Clear() const;

  /**
   * @brief Close the controller
   * @return return Ok(whether succeeded to close), or Err(error msg) if some unrecoverable error occurred
   */
  [[nodiscard]] Error Close();

  /**
   * @brief Stop outputting
   * @return return Ok(whether succeeded to stop), or Err(error msg) if some unrecoverable error occurred
   */
  [[nodiscard]] Error Stop();

  /**
   * @brief Send gain to the device
   * @param[in] gain Gain to display
   * @param[in] wait_for_sent if true, this function will wait for the data is sent to the devices and processed.
   * @return return Ok(whether succeeded), or Err(error msg) if some unrecoverable error occurred
   */
  [[nodiscard]] Error Send(const core::GainPtr& gain, bool wait_for_sent = false);

  /**
   * @brief Send modulation to the device
   * @param[in] mod Amplitude modulation to display
   * @return return Ok(whether succeeded), or Err(error msg) if some unrecoverable error occurred
   */
  [[nodiscard]] Error Send(const core::ModulationPtr& mod);

  /**
   * @brief Send gain and modulation to the device
   * @param[in] gain Gain to display
   * @param[in] mod Amplitude modulation to display
   * @param[in] wait_for_sent if true, this function will wait for the data is sent to the devices and processed.
   * @return return Ok(whether succeeded), or Err(error msg) if some unrecoverable error occurred
   */
  [[nodiscard]] Error Send(const core::GainPtr& gain, const core::ModulationPtr& mod, bool wait_for_sent = false);

  /**
   * @brief Send sequence and modulation to the device
   * @param[in] seq Sequence to display
   * @return return Ok(whether succeeded), or Err(error msg) if some unrecoverable error occurred
   */
  [[nodiscard]] Error Send(const core::SequencePtr& seq);

  /**
   * @brief Enumerate firmware information
   * @return return Ok(firmware_info_list), or Err(error msg) if some unrecoverable error occurred
   */
  [[nodiscard]] Result<std::vector<FirmwareInfo>, std::string> firmware_info_list() const;

  [[nodiscard]] std::shared_ptr<STMController> stm() const;

  class STMController {
   public:
    explicit STMController(core::LinkPtr link, core::GeometryPtr geometry, bool* silent_mode, bool* read_fpga_info)
        : _link(std::move(link)), _geometry(std::move(geometry)), _silent_mode(silent_mode), _read_fpga_info(read_fpga_info) {}

    /**
     * @brief Add gain for STM
     */
    void AddGain(const core::GainPtr& gain);

    /**
     * @brief Add gains for STM
     */
    void AddGains(const std::vector<core::GainPtr>& gains);

    /**
     * @brief Start Spatio-Temporal Modulation
     * @param[in] freq Frequency of STM modulation
     * @details Generate STM modulation by switching gains appended by
     * AddSTMGain() at the freq. The accuracy depends on the computer, for
     * example, about 1ms on Windows. Note that it is affected by interruptions,
     * and so on.
     * @return return Ok(whether succeeded), or Err(error msg) if some unrecoverable error occurred
     */
    [[nodiscard]] Error Start(double freq);

    /**
     * @brief Suspend Spatio-Temporal Modulation
     * @return return Ok(whether succeeded), or Err(error msg) if some unrecoverable error occurred
     */
    [[nodiscard]] Error Stop();

    /**
     * @brief Finish Spatio-Temporal Modulation
     * @details Appended gains will be removed.
     * @return return Ok(whether succeeded), or Err(error msg) if some unrecoverable error occurred
     */
    [[nodiscard]] Error Finish();

   private:
    core::LinkPtr _link;
    core::GeometryPtr _geometry;
    std::vector<core::GainPtr> _gains;
    std::vector<std::unique_ptr<uint8_t[]>> _bodies;
    std::vector<size_t> _sizes;
    Timer _timer;
    std::atomic<bool> _lock;
    bool* _silent_mode;
    bool* _read_fpga_info;
  };

 private:
  [[nodiscard]] Error SendHeader(core::COMMAND cmd, size_t max_trial = 50) const;
  [[nodiscard]] Error WaitMsgProcessed(uint8_t msg_id, size_t max_trial = 200) const;

  core::LinkPtr _link;
  core::GeometryPtr _geometry;
  bool _silent_mode;
  bool _read_fpga_info;
  bool _seq_mode;
  core::Configuration _config;

  std::unique_ptr<uint8_t[]> _tx_buf;
  std::unique_ptr<uint8_t[]> _rx_buf;
  std::vector<uint8_t> _fpga_infos;

  std::shared_ptr<STMController> _stm;
};
}  // namespace autd
