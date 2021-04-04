// File: controller.hpp
// Project: include
// Created Date: 11/04/2018
// Author: Shun Suzuki
// -----
// Last Modified: 04/04/2021
// Modified By: Shun Suzuki (suzuki@hapis.k.u-tokyo.ac.jp)
// -----
// Copyright (c) 2018-2020 Hapis Lab. All rights reserved.
//

#pragma once

#include <memory>
#include <string>
#include <vector>

#include "configuration.hpp"
#include "firmware_version.hpp"
#include "gain.hpp"
#include "geometry.hpp"
#include "link.hpp"
#include "modulation.hpp"
#include "sequence.hpp"

namespace autd {

class Controller;
using ControllerPtr = std::unique_ptr<Controller>;

/**
 * @brief AUTD Controller
 */
class Controller {
 public:
  Controller() noexcept = default;
  virtual ~Controller() = default;
  Controller(const Controller& v) noexcept = delete;
  Controller& operator=(const Controller& obj) = delete;
  Controller(Controller&& obj) = delete;
  Controller& operator=(Controller&& obj) = delete;

  /**
   * @brief Create controller
   */
  static ControllerPtr Create();
  /**
   * @brief Verify that the device is properly connected
   */
  virtual bool is_open() = 0;
  /**
   * @brief Geometry of the devices
   */
  virtual GeometryPtr geometry() noexcept = 0;
  /**
   * @brief Check silent mode
   */
  virtual bool silent_mode() noexcept = 0;
  /**
   * @brief Count the remaining data (Gain and Modulation) in buffer
   */
  virtual size_t remaining_in_buffer() = 0;

  /**
   * @brief Open device with a specific link.
   * @param[in] link Link
   * @return return Ok(whether succeeded to open), or Err(error msg) if some unrecoverable error occurred
   */
  [[nodiscard]] virtual Result<bool, std::string> OpenWith(LinkPtr link) = 0;

  /**
   * @brief Set silent mode
   */
  virtual void SetSilentMode(bool silent) noexcept = 0;

  /**
   * @brief Synchronize all devices
   * @details Call this function only once after OpenWith(). It takes several seconds and blocks the thread in the meantime.
   * @param[in] config configuration
   * @return return Ok(whether succeeded to synchronize), or Err(error msg) if some unrecoverable error occurred
   */
  [[nodiscard]] virtual Result<bool, std::string> Synchronize(Configuration config = Configuration::GetDefaultConfiguration()) = 0;

  /**
   * @brief Clear all data in hardware
   * @return return Ok(whether succeeded to clear), or Err(error msg) if some unrecoverable error occurred
   */
  [[nodiscard]] virtual Result<bool, std::string> Clear() = 0;

  /**
   * @brief Close the controller
   * @return return Ok(whether succeeded to close), or Err(error msg) if some unrecoverable error occurred
   */
  [[nodiscard]] virtual Result<bool, std::string> Close() = 0;

  /**
   * @brief Stop outputting
   * @return return Ok(whether succeeded to stop), or Err(error msg) if some unrecoverable error occurred
   */
  [[nodiscard]] virtual Result<bool, std::string> Stop() = 0;
  /**
   * @brief Append gain to the controller (non blocking)
   * @param[in] gain Gain to display
   * @details Gain will be sent in another thread
   * @return return Ok(whether succeeded), or Err(error msg) if some unrecoverable error occurred
   */
  [[nodiscard]] virtual Result<bool, std::string> AppendGain(GainPtr gain) = 0;
  /**
   * @brief Append gain to the controller (blocking)
   * @param[in] gain Gain to display
   * @param[in] wait_for_send if true, wait for the data to arrive on devices by handshaking
   * @details Gain will be build in this function.
   * @return return Ok(whether succeeded), or Err(error msg) if some unrecoverable error occurred
   */
  [[nodiscard]] virtual Result<bool, std::string> AppendGainSync(GainPtr gain, bool wait_for_send = false) = 0;
  /**
   * @brief Append modulation to the controller (non blocking)
   * @details Modulation will be sent in another thread
   * @return return Ok(whether succeeded), or Err(error msg) if some unrecoverable error occurred
   */
  [[nodiscard]] virtual Result<bool, std::string> AppendModulation(ModulationPtr modulation) = 0;
  /**
   * @brief Append modulation to the controller (blocking)
   * @return return Ok(whether succeeded), or Err(error msg) if some unrecoverable error occurred
   */
  [[nodiscard]] virtual Result<bool, std::string> AppendModulationSync(ModulationPtr modulation) = 0;

  /**
   * @brief Add gain for STM
   */
  virtual void AddSTMGain(GainPtr gain) = 0;
  /**
   * @brief Add gain for STM
   */
  virtual void AddSTMGain(const std::vector<GainPtr>& gain_list) = 0;

  /**
   * @brief Start Spatio-Temporal Modulation
   * @param[in] freq Frequency of STM modulation
   * @details Generate STM modulation by switching gains appended by
   * AddSTMGain() at the freq. The accuracy depends on the computer, for
   * example, about 1ms on Windows. Note that it is affected by interruptions,
   * and so on.
   * @return return Ok(whether succeeded), or Err(error msg) if some unrecoverable error occurred
   */
  [[nodiscard]] virtual Result<bool, std::string> StartSTModulation(Float freq) = 0;

  /**
   * @brief Suspend Spatio-Temporal Modulation
   * @return return Ok(whether succeeded), or Err(error msg) if some unrecoverable error occurred
   */
  [[nodiscard]] virtual Result<bool, std::string> StopSTModulation() = 0;

  /**
   * @brief Finish Spatio-Temporal Modulation
   * @details Appended gains will be removed.
   * @return return Ok(whether succeeded), or Err(error msg) if some unrecoverable error occurred
   */
  [[nodiscard]] virtual Result<bool, std::string> FinishSTModulation() = 0;

  /**
   * @brief Append sequence to the controller (blocking)
   * @return return Ok(whether succeeded), or Err(error msg) if some unrecoverable error occurred
   */
  [[nodiscard]] virtual Result<bool, std::string> AppendSequence(SequencePtr seq) = 0;

  /**
   * @brief Flush the buffer
   */
  virtual void Flush() = 0;
  /**
   * @brief Enumerate firmware information
   * @return return Ok(firmware_info_list), or Err(error msg) if some unrecoverable error occurred
   */
  [[nodiscard]] virtual Result<std::vector<FirmwareInfo>, std::string> firmware_info_list() = 0;
};
}  // namespace autd
