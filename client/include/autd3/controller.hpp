﻿// File: controller.hpp
// Project: include
// Created Date: 05/11/2020
// Author: Shun Suzuki
// -----
// Last Modified: 20/07/2021
// Modified By: Shun Suzuki (suzuki@hapis.k.u-tokyo.ac.jp)
// -----
// Copyright (c) 2021 Hapis Lab. All rights reserved.
//

#pragma once

#include <memory>
#include <utility>
#include <vector>

#include "core/firmware_version.hpp"
#include "core/gain.hpp"
#include "core/geometry.hpp"
#include "core/link.hpp"
#include "core/logic.hpp"
#include "core/modulation.hpp"
#include "core/osal_timer.hpp"
#include "core/sequence.hpp"

namespace autd {

class Controller;
using ControllerPtr = std::unique_ptr<Controller>;

/**
 * @brief AUTD Controller
 */
class Controller {
  class STMTimerCallback;

  struct ControllerProps {
    friend class Controller;
    friend class STMController;
    ControllerProps(const bool silent_mode, const bool reads_fpga_info, const bool seq_mode, const bool force_fan)
        : _silent_mode(silent_mode), _reads_fpga_info(reads_fpga_info), _seq_mode(seq_mode), _force_fan(force_fan) {}
    ~ControllerProps() = default;
    ControllerProps(const ControllerProps& v) noexcept = delete;
    ControllerProps& operator=(const ControllerProps& obj) = delete;
    ControllerProps(ControllerProps&& obj) = default;
    ControllerProps& operator=(ControllerProps&& obj) = default;

   private:
    [[nodiscard]] uint8_t ctrl_flag() const;

    bool _silent_mode;
    bool _reads_fpga_info;
    bool _seq_mode;
    bool _force_fan;
  };

 public:
  class STMController;
  class STMTimer;

  static ControllerPtr create();

  /**
   * @brief Verify the device is properly connected
   */
  [[nodiscard]] bool is_open() const;

  /**
   * @brief Geometry of the devices
   */
  [[nodiscard]] core::GeometryPtr& geometry() noexcept;

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
  std::vector<uint8_t> fpga_info();

  /**
   * @brief Update control flag
   */
  bool update_ctrl_flag();

  /**
   * \brief Set output delay
   * \param[in] delay delay for each transducer in units of ultrasound period (i.e. 25us).
   * \return if true, It guarantees that the devices have processed the data
   */
  bool set_output_delay(const std::vector<std::array<uint8_t, core::NUM_TRANS_IN_UNIT>>& delay);

  /**
   * \brief Set duty offset
   * \param[in] offset duty offset for each transducers (only the lowest 1 bit will be used)
   * \return if true, It guarantees that the devices have processed the data
   */
  bool set_duty_offset(const std::vector<std::array<uint8_t, core::NUM_TRANS_IN_UNIT>>& offset);

  /**
   * \brief Set delay and duty offset
   * \param[in] delay delay
   * \param[in] offset duty offset
   * \return if true, It guarantees that the devices have processed the data
   */
  bool set_delay_offset(const std::vector<std::array<uint8_t, core::NUM_TRANS_IN_UNIT>>& delay,
                        const std::vector<std::array<uint8_t, core::NUM_TRANS_IN_UNIT>>& offset);

  /**
   * @brief Open device with a link.
   * @param[in] link Link
   */
  void open(core::LinkPtr link);

  /**
   * @brief Clear all data in hardware
   * \return if true, It guarantees that the devices have processed the data
   */
  bool clear();

  /**
   * @brief Close the controller
   * \return if true, It guarantees that the devices have processed the data
   */
  bool close();

  /**
   * @brief Stop outputting
   * \return if true, It guarantees that the devices have processed the data
   */
  bool stop();

  /**
   * @brief Pause outputting
   * \return if true, It guarantees that the devices have processed the data
   */
  bool pause() const;

  /**
   * @brief Resume outputting
   * \return if true, It guarantees that the devices have processed the data
   */
  bool resume() const;

  /**
   * @brief Send gain to the device
   * @param[in] gain Gain to display
   * \return if true, It guarantees that the devices have processed the data
   */
  bool send(const core::GainPtr& gain);

  /**
   * @brief Send modulation to the device
   * @param[in] mod Amplitude modulation to display
   * \return if true, It guarantees that the devices have processed the data
   */
  bool send(const core::ModulationPtr& mod);

  /**
   * @brief Send gain and modulation to the device
   * @param[in] gain Gain to display
   * @param[in] mod Amplitude modulation to display
   * \return if true, It guarantees that the devices have processed the data
   */
  bool send(const core::GainPtr& gain, const core::ModulationPtr& mod);

  /**
   * @brief Send sequence to the device
   * @param[in] seq Sequence to display
   * \return if true, It guarantees that the devices have processed the data
   */
  bool send(const core::PointSequencePtr& seq);

  /**
   * @brief Send sequence to the device
   * @param[in] seq Sequence to display
   * \return if true, It guarantees that the devices have processed the data
   */
  bool send(const core::GainSequencePtr& seq);

  /**
   * @brief Enumerate firmware information
   * \return firmware information list. If failed, the vector is empty.
   */
  [[nodiscard]] std::vector<FirmwareInfo> firmware_info_list() const;

  /**
   * \brief return pointer to software spatio-temporal modulation controller.
   * \details Never use Controller before calling STMController::finish
   */
  [[nodiscard]] std::unique_ptr<STMController> stm();

  /**
   * \brief Software spatio-temporal modulation controller.
   */
  class STMController {
   public:
    friend class Controller;

    /**
     * @brief Add gain for STM
     */
    void add_gain(const core::GainPtr& gain) const;

    /**
     * @brief Start Spatio-Temporal Modulation
     * @param[in] freq Frequency of STM modulation
     * @details Generate STM modulation by switching gains appended by
     * add_gain() at the freq. The accuracy depends on the computer, for
     * example, about 1ms on Windows. Note that it is affected by interruptions,
     * and so on.
     */
    void start(double freq);

    /**
     * \brief Suspend Spatio-Temporal Modulation
     */
    void stop();

    /**
     * @brief Finish Spatio-Temporal Modulation
     * @details Added gains will be removed. Never use this STMController after calling this function.
     */
    void finish();

   private:
    explicit STMController(Controller* p_cnt, std::unique_ptr<STMTimerCallback> handler)
        : _p_cnt(p_cnt), _handler(std::move(handler)), _timer(nullptr) {}

    Controller* _p_cnt;
    std::unique_ptr<STMTimerCallback> _handler;
    std::unique_ptr<core::Timer<STMTimerCallback>> _timer;
  };

 private:
  Controller() noexcept
      : _link(nullptr),
        _geometry(std::make_unique<core::Geometry>()),
        _props(ControllerProps(true, false, false, false)),
        _tx_buf(nullptr),
        _rx_buf(nullptr) {}

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
        this->_link->send(&this->_bodies[_idx][0], this->_sizes[_idx]);
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

  [[nodiscard]] bool send_header(core::COMMAND cmd) const;
  void init_delay_offset();
  [[nodiscard]] bool send_delay_offset() const;
  [[nodiscard]] bool wait_msg_processed(uint8_t msg_id, size_t max_trial = 50) const;

  core::LinkPtr _link;
  core::GeometryPtr _geometry;
  ControllerProps _props;
  std::unique_ptr<uint8_t[]> _tx_buf;
  std::unique_ptr<uint8_t[]> _rx_buf;

  std::vector<uint8_t> _fpga_infos;
  std::vector<std::array<uint8_t, core::NUM_TRANS_IN_UNIT>> _delay;
  std::vector<std::array<uint8_t, core::NUM_TRANS_IN_UNIT>> _offset;
};
}  // namespace autd
