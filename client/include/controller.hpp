// File: controller.hpp
// Project: include
// Created Date: 11/04/2018
// Author: Shun Suzuki
// -----
// Last Modified: 01/11/2020
// Modified By: Shun Suzuki (suzuki@hapis.k.u-tokyo.ac.jp)
// -----
// Copyright (c) 2018-2020 Hapis Lab. All rights reserved.
//

#pragma once

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "configuration.hpp"
#include "gain.hpp"
#include "geometry.hpp"
#include "modulation.hpp"
#include "vector3.hpp"

namespace autd {

enum class AUTD_VERSION {
  V_0_1 = 0,
  V_0_6 = 1,
  V_0_7 = 2,
};

/**
 * @brief AUTD Controller
 */
class Controller {
 public:
  virtual ~Controller() {}

  /**
   * @brief Create controller
   */
  static ControllerPtr Create(AUTD_VERSION version = AUTD_VERSION::V_0_7);
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
   */
  virtual void OpenWith(LinkPtr link) = 0;

  /**
   * @brief Set silent mode
   */
  virtual void SetSilentMode(bool silent) noexcept = 0;
  /**
   * @brief Calibrate
   * @details Call this function only once after OpenWith(). It takes several seconds and blocks the thread in the meantime.
   * @param[in] config configration
   * @return true if success to calibrate
   */
  virtual bool Calibrate(Configuration config = Configuration::GetDefaultConfiguration()) = 0;
  /**
   * @brief Clear all data in hardware
   * @return true if success to clear
   */
  virtual bool Clear() = 0;
  /**
   * @brief Close the controller
   */
  virtual void Close() = 0;

  /**
   * @brief Stop outputting
   */
  virtual void Stop() = 0;
  /**
   * @brief Append gain to the controller (non blocking)
   * @param[in] gain Gain to display
   * @details Gain will be sent in another thread
   */
  virtual void AppendGain(GainPtr gain) = 0;
  /**
   * @brief Append gain to the controller (blocking)
   * @param[in] gain Gain to display
   * @param[in] wait_for_send if true, wait for the data to arrive on devices by handshaking
   * @details Gain will be build in this function.
   */
  virtual void AppendGainSync(GainPtr gain, bool wait_for_send = false) = 0;
  /**
   * @brief Append modulation to the controller (non blocking)
   * @details Modulation will be sent in another thread
   */
  virtual void AppendModulation(ModulationPtr modulation) = 0;
  /**
   * @brief Append modulation to the controller (blocking)
   */
  virtual void AppendModulationSync(ModulationPtr modulation) = 0;
  /**
   * @brief Append gain for STM
   */
  virtual void AppendSTMGain(GainPtr gain) = 0;
  /**
   * @brief Append gain for STM
   */
  virtual void AppendSTMGain(const std::vector<GainPtr> &gain_list) = 0;
  /**
   * @brief Start Spatio-Temporal Modulation
   * @param[in] freq Frequency of STM modulation
   * @details Generate STM modulation by switching gains appended by
   * AppendSTMGain() at the freq. The accuracy depends on the computer, for
   * example, about 1ms on Windows. Note that it is affected by interruptions,
   * and so on.
   */
  virtual void StartSTModulation(double freq) = 0;
  /**
   * @brief Suspend Spatio-Temporal Modulation
   */
  virtual void StopSTModulation() = 0;
  /**
   * @brief Finish Spatio-Temporal Modulation
   * @details Appended gains will be removed.
   */
  virtual void FinishSTModulation() = 0;

  /**
   * @brief Append sequence to the controller (blocking)
   */
  virtual void AppendSequence(SequencePtr seq) = 0;

  /**
   * @brief Flush the buffer
   */
  virtual void Flush() = 0;
  /**
   * @brief Enumerate firmware infomations
   */
  virtual FirmwareInfoList firmware_info_list() = 0;

  virtual void LateralModulationAT(Vector3 point, Vector3 dir = Vector3::unit_y(), double lm_amp = 2.5, double lm_freq = 100) = 0;
};
}  // namespace autd
