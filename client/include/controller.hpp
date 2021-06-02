// File: controller.hpp
// Project: include
// Created Date: 05/11/2020
// Author: Shun Suzuki
// -----
// Last Modified: 02/06/2021
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
  class STMTimerCallback;

  struct ControllerProps {
    friend class Controller;
    friend class STMController;
    ControllerProps(const core::Configuration config, core::GeometryPtr geometry, const bool silent_mode, const bool reads_fpga_info,
                    const bool seq_mode, const bool force_fan, std::unique_ptr<uint8_t[]> tx_buf, std::unique_ptr<uint8_t[]> rx_buf)
        : _config(config),
          _geometry(std::move(geometry)),
          _silent_mode(silent_mode),
          _reads_fpga_info(reads_fpga_info),
          _seq_mode(seq_mode),
          _force_fan(force_fan),
          _tx_buf(std::move(tx_buf)),
          _rx_buf(std::move(rx_buf)) {}
    ~ControllerProps() = default;
    ControllerProps(const ControllerProps& v) noexcept = delete;
    ControllerProps& operator=(const ControllerProps& obj) = delete;
    ControllerProps(ControllerProps&& obj) = default;
    ControllerProps& operator=(ControllerProps&& obj) = default;

   private:
    [[nodiscard]] uint8_t ctrl_flag() const;

    core::Configuration _config;
    core::GeometryPtr _geometry;
    bool _silent_mode;
    bool _reads_fpga_info;
    bool _seq_mode;
    bool _force_fan;
    std::unique_ptr<uint8_t[]> _tx_buf;
    std::unique_ptr<uint8_t[]> _rx_buf;
  };

 public:
  class STMController;
  class STMTimer;

  static std::unique_ptr<Controller> create();

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
   * @brief If true, the fan will be forced to start.
   */
  bool& force_fan() noexcept;

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
   * \return ok(whether succeeded), or err(error message) if unrecoverable error is occurred
   */
  [[nodiscard]] Error send(const core::GainPtr& gain);

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
   * \return ok(whether succeeded), or err(error message) if unrecoverable error is occurred
   */
  [[nodiscard]] Error send(const core::GainPtr& gain, const core::ModulationPtr& mod);

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
  [[nodiscard]] std::unique_ptr<STMController> stm();

  /**
   * \brief Software spatio-temporal modulation controller.
   */
  class STMController {
   public:
    friend class Controller;

    /**
     * @brief Return controller
     */
    [[nodiscard]] std::unique_ptr<Controller> controller();

    /**
     * @brief Add gain for STM
     */
    [[nodiscard]] Error add_gain(const core::GainPtr& gain) const;

    /**
     * @brief Start Spatio-Temporal Modulation
     * @param[in] freq Frequency of STM modulation
     * @details Generate STM modulation by switching gains appended by
     * add_gain() at the freq. The accuracy depends on the computer, for
     * example, about 1ms on Windows. Note that it is affected by interruptions,
     * and so on.
     * \return ok with STMTimer, or err(error message) if unrecoverable error is occurred
     */
    [[nodiscard]] Result<std::unique_ptr<STMTimer>, std::string> start(double freq);

    /**
     * @brief Finish Spatio-Temporal Modulation
     * @details Added gains will be removed.
     */
    void finish() const;

   private:
    explicit STMController(std::unique_ptr<STMTimerCallback> handler, ControllerProps props)
        : _props(std::move(props)), _handler(std::move(handler)) {}

    ControllerProps _props;
    std::unique_ptr<STMTimerCallback> _handler;
  };

  /**
   * \brief Software spatio-temporal modulation timer.
   */
  class STMTimer {
   public:
    friend class STMController;

    /**
     * @brief Suspend Spatio-Temporal Modulation
     * \return ok with STMController, or err(error message) if unrecoverable error is occurred
     */
    [[nodiscard]] Result<std::unique_ptr<STMController>, std::string> stop();

   private:
    explicit STMTimer(std::unique_ptr<core::Timer<STMTimerCallback>> timer, ControllerProps props)
        : _timer(std::move(timer)), _props(std::move(props)) {}

    std::unique_ptr<core::Timer<STMTimerCallback>> _timer;
    ControllerProps _props;
  };

 private:
  Controller() noexcept
      : _link(nullptr),
        _props(ControllerProps(core::Configuration::get_default_configuration(), std::make_shared<core::Geometry>(), true, false, false, false,
                               nullptr, nullptr)) {}

  explicit Controller(core::LinkPtr link, ControllerProps props) noexcept : _link(std::move(link)), _props(std::move(props)) {}

  class STMTimerCallback final : core::CallbackHandler {
   public:
    friend class STMController;

    virtual ~STMTimerCallback() = default;
    STMTimerCallback(const STMTimerCallback& v) noexcept = delete;
    STMTimerCallback& operator=(const STMTimerCallback& obj) = delete;
    STMTimerCallback(STMTimerCallback&& obj) = delete;
    STMTimerCallback& operator=(STMTimerCallback&& obj) = delete;

    explicit STMTimerCallback(core::LinkPtr link) : _link(std::move(link)), _idx(0), _lock(false) {}

    void add(std::unique_ptr<uint8_t[]> data, const size_t size) {
      this->_bodies.emplace_back(std::move(data));
      this->_sizes.emplace_back(size);
    }
    void clear() {
      std::vector<std::unique_ptr<uint8_t[]>>().swap(this->_bodies);
      std::vector<size_t>().swap(this->_sizes);
      this->_idx = 0;
    }

    void callback() override {
      if (auto expected = false; _lock.compare_exchange_weak(expected, true)) {
        this->_link->send(&this->_bodies[_idx][0], this->_sizes[_idx]).unwrap();
        this->_idx = (this->_idx + 1) % this->_bodies.size();
        _lock.store(false, std::memory_order_release);
      }
    }

   private:
    core::LinkPtr _link;
    std::vector<std::unique_ptr<uint8_t[]>> _bodies;
    std::vector<size_t> _sizes;
    size_t _idx;
    std::atomic<bool> _lock;
  };

  [[nodiscard]] Error send_header(core::COMMAND cmd) const;
  [[nodiscard]] Error wait_msg_processed(uint8_t msg_id, size_t max_trial = 50) const;

  core::LinkPtr _link;
  ControllerProps _props;

  std::vector<uint8_t> _fpga_infos;
};
}  // namespace autd
