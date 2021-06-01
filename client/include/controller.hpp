// File: controller.hpp
// Project: include
// Created Date: 05/11/2020
// Author: Shun Suzuki
// -----
// Last Modified: 01/06/2021
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
        _config(core::Configuration::get_default_configuration()),
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
   *  \return ok with FPGA information if succeeded, or err with error message if failed
   *  \details the first bit of FPGA info represents whether the fan is running.
   */
  Result<std::vector<uint8_t>, std::string> fpga_info();

  /**
   * @brief Update control flag
   */
  Error update_ctrl_flag();

  /**
   * \brief Set output delay
   * \param[in] delay delay for each transducer in units of ultrasound period (i.e. 25us).
   * \return ok(whether succeeded), or err(error message) if unrecoverable error is occurred
   * \details The maximum value of delay is 128. If you set a value of more than 128, the lowest 7 bits will be used.
   */
  [[nodiscard]] Error set_output_delay(const std::vector<core::DataArray>& delay) const;

  /**
   * @brief Open device with a link.
   * @param[in] link Link
   * \return ok(whether succeeded), or err(error message) if unrecoverable error is occurred
   */
  [[nodiscard]] Error open(core::LinkPtr link);

  /**
   * @brief Synchronize all devices
   * @param[in] config configuration
   * \return ok(whether succeeded), or err(error message) if unrecoverable error is occurred
   */
  [[nodiscard]] Error synchronize(core::Configuration config = core::Configuration::get_default_configuration());

  /**
   * @brief Clear all data in hardware
   * \return ok(whether succeeded), or err(error message) if unrecoverable error is occurred
   */
  [[nodiscard]] Error clear() const;

  /**
   * @brief Close the controller
   * \return ok(whether succeeded), or err(error message) if unrecoverable error is occurred
   */
  [[nodiscard]] Error close();

  /**
   * @brief Stop outputting
   * \return ok(whether succeeded), or err(error message) if unrecoverable error is occurred
   */
  [[nodiscard]] Error stop();

  /**
   * @brief Send gain to the device
   * @param[in] gain Gain to display
   * @param[in] wait_for_sent if true, this function will wait for the data is sent to the devices and processed.
   * \return ok(whether succeeded), or err(error message) if unrecoverable error is occurred
   */
  [[nodiscard]] Error send(const core::GainPtr& gain, bool wait_for_sent = false);

  /**
   * @brief Send modulation to the device
   * @param[in] mod Amplitude modulation to display
   * \return ok(whether succeeded), or err(error message) if unrecoverable error is occurred
   */
  [[nodiscard]] Error send(const core::ModulationPtr& mod);

  /**
   * @brief Send gain and modulation to the device
   * @param[in] gain Gain to display
   * @param[in] mod Amplitude modulation to display
   * @param[in] wait_for_sent if true, this function will wait for the data is sent to the devices and processed.
   * \return ok(whether succeeded), or err(error message) if unrecoverable error is occurred
   */
  [[nodiscard]] Error send(const core::GainPtr& gain, const core::ModulationPtr& mod, bool wait_for_sent = false);

  /**
   * @brief Send sequence and modulation to the device
   * @param[in] seq Sequence to display
   * \return ok(whether succeeded), or err(error message) if unrecoverable error is occurred
   */
  [[nodiscard]] Error send(const core::SequencePtr& seq);

  /**
   * @brief Enumerate firmware information
   * \return ok with firmware information list if succeeded, or err with error message if failed
   */
  [[nodiscard]] Result<std::vector<FirmwareInfo>, std::string> firmware_info_list() const;

  /**
   * \brief return pointer to software spatio-temporal modulation controller.
   */
  [[nodiscard]] std::shared_ptr<STMController> stm() const;

  /**
   * \brief Software spatio-temporal modulation controller.
   */
  class STMController {
   public:
    explicit STMController(core::LinkPtr link, core::GeometryPtr geometry, bool* silent_mode, bool* read_fpga_info)
        : _link(std::move(link)), _geometry(std::move(geometry)), _silent_mode(silent_mode), _read_fpga_info(read_fpga_info) {}

    /**
     * @brief Add gain for STM
     */
    void add_gain(const core::GainPtr& gain);

    /**
     * @brief Add gains for STM
     */
    void add_gains(const std::vector<core::GainPtr>& gains);

    /**
     * @brief Start Spatio-Temporal Modulation
     * @param[in] freq Frequency of STM modulation
     * @details Generate STM modulation by switching gains appended by
     * add_gain() or add_gains() at the freq. The accuracy depends on the computer, for
     * example, about 1ms on Windows. Note that it is affected by interruptions,
     * and so on.
     * \return ok(whether succeeded), or err(error message) if unrecoverable error is occurred
     */
    [[nodiscard]] Error start(double freq);

    /**
     * @brief Suspend Spatio-Temporal Modulation
     * \return ok(whether succeeded), or err(error message) if unrecoverable error is occurred
     */
    [[nodiscard]] Error stop();

    /**
     * @brief Finish Spatio-Temporal Modulation
     * @details Added gains will be removed.
     * \return ok(whether succeeded), or err(error message) if unrecoverable error is occurred
     */
    [[nodiscard]] Error finish();

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
  [[nodiscard]] Error send_header(core::COMMAND cmd, size_t max_trial = 50) const;
  [[nodiscard]] Error wait_msg_processed(uint8_t msg_id, size_t max_trial = 200) const;

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
