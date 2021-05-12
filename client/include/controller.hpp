// File: controller.hpp
// Project: include
// Created Date: 05/11/2020
// Author: Shun Suzuki
// -----
// Last Modified: 12/05/2021
// Modified By: Shun Suzuki (suzuki@hapis.k.u-tokyo.ac.jp)
// -----
// Copyright (c) 2021 Hapis Lab. All rights reserved.
//

#pragma once

#include <memory>
#include <string>
#include <vector>

#include "core/configuration.hpp"
#include "core/firmware_version.hpp"
#include "core/gain.hpp"
#include "core/geometry.hpp"
#include "core/link.hpp"
#include "core/logic.hpp"
#include "core/modulation.hpp"
#include "core/osal_timer.hpp"

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
        _seq_mode(false),
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
   * @brief Open device with a link.
   * @param[in] link Link
   * @return return Ok(whether succeeded to open), or Err(error msg) if some unrecoverable error occurred
   */
  [[nodiscard]] Result<bool, std::string> OpenWith(core::LinkPtr link);

  /**
   * @brief Synchronize all devices
   * @details Call this function only once after OpenWith(). It takes several seconds and blocks the thread in the meantime.
   * @param[in] config configuration
   * @return return Ok(whether succeeded to synchronize), or Err(error msg) if some unrecoverable error occurred
   */
  [[nodiscard]] Result<bool, std::string> Synchronize(core::Configuration config = core::Configuration::GetDefaultConfiguration());

  /**
   * @brief Clear all data in hardware
   * @return return Ok(whether succeeded to clear), or Err(error msg) if some unrecoverable error occurred
   */
  [[nodiscard]] Result<bool, std::string> Clear() const;

  /**
   * @brief Close the controller
   * @return return Ok(whether succeeded to close), or Err(error msg) if some unrecoverable error occurred
   */
  [[nodiscard]] Result<bool, std::string> Close();

  /**
   * @brief Stop outputting
   * @return return Ok(whether succeeded to stop), or Err(error msg) if some unrecoverable error occurred
   */
  [[nodiscard]] Result<bool, std::string> Stop() const;

  /**
   * @brief Send gain to the controller
   * @param[in] gain Gain to display
   * @param[in] mod Amplitude modulation to display
   * @param[in] wait_for_sent if true, this function will wait for the data is sent to the devices and processed.
   * @return return Ok(whether succeeded), or Err(error msg) if some unrecoverable error occurred
   */
  [[nodiscard]] Result<bool, std::string> Send(core::GainPtr gain, core::ModulationPtr mod, bool wait_for_sent = false) const;

  /**
   * @brief Enumerate firmware information
   * @return return Ok(firmware_info_list), or Err(error msg) if some unrecoverable error occurred
   */
  [[nodiscard]] Result<std::vector<FirmwareInfo>, std::string> firmware_info_list() const;

  [[nodiscard]] std::shared_ptr<STMController> stm() const;

  class STMController {
   public:
    explicit STMController(const core::LinkPtr link, const core::GeometryPtr geometry, bool* silent_mode)
        : _link(link), _geometry(geometry), _silent_mode(silent_mode) {}

    /**
     * @brief Add gain for STM
     */
    void AddGain(core::GainPtr gain);

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
    [[nodiscard]] Result<bool, std::string> Start(double freq);

    /**
     * @brief Suspend Spatio-Temporal Modulation
     * @return return Ok(whether succeeded), or Err(error msg) if some unrecoverable error occurred
     */
    [[nodiscard]] Result<bool, std::string> Stop();

    /**
     * @brief Finish Spatio-Temporal Modulation
     * @details Appended gains will be removed.
     * @return return Ok(whether succeeded), or Err(error msg) if some unrecoverable error occurred
     */
    [[nodiscard]] Result<bool, std::string> Finish();

   private:
    core::LinkPtr _link;
    core::GeometryPtr _geometry;
    std::vector<core::GainPtr> _gains;
    std::vector<uint8_t*> _bodies;
    std::vector<size_t> _body_sizes;
    Timer _timer;
    std::atomic<bool> _lock;
    bool* _silent_mode;
  };

 private:
  core::LinkPtr _link;
  core::GeometryPtr _geometry;
  bool _silent_mode;
  bool _seq_mode;
  core::Configuration _config;

  std::unique_ptr<uint8_t[]> _tx_buf;
  std::unique_ptr<uint8_t[]> _rx_buf;

  std::shared_ptr<STMController> _stm;
};
}  // namespace autd
