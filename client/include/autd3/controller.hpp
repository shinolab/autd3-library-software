// File: controller.hpp
// Project: include
// Created Date: 05/11/2020
// Author: Shun Suzuki
// -----
// Last Modified: 12/12/2021
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
#include "core/interface.hpp"
#include "core/link.hpp"
#include "core/osal_timer.hpp"

namespace autd {

/**
 * @brief AUTD Controller
 */
class Controller {
  class STMTimerCallback;

  struct ControllerProps {
    friend class Controller;
    friend class STMController;
    ControllerProps() : _output_enable(false), _output_balance(false), _silent_mode(true), _force_fan(false), _reads_fpga_info(false) {}
    ~ControllerProps() = default;
    ControllerProps(const ControllerProps& v) noexcept = delete;
    ControllerProps& operator=(const ControllerProps& obj) = delete;
    ControllerProps(ControllerProps&& obj) = default;
    ControllerProps& operator=(ControllerProps&& obj) = default;

   private:
    [[nodiscard]] uint8_t fpga_ctrl_flag() const;
    [[nodiscard]] uint8_t cpu_ctrl_flag() const;

    bool _output_enable;
    bool _output_balance;
    bool _silent_mode;
    bool _force_fan;
    bool _reads_fpga_info;
  };

 public:
  class STMController;
  class STMTimer;

  Controller() noexcept : _link(nullptr), _props(ControllerProps()), _check_ack(false) {}
  ~Controller() noexcept;
  Controller(const Controller& v) noexcept = delete;
  Controller& operator=(const Controller& obj) = delete;
  Controller(Controller&& obj) = default;
  Controller& operator=(Controller&& obj) = default;

  /**
   * @brief Verify the device is properly connected
   */
  [[nodiscard]] bool is_open() const;

  /**
   * @brief Geometry of the devices
   */
  [[nodiscard]] core::Geometry& geometry() noexcept;

  /**
   * @brief Geometry of the devices
   */
  [[nodiscard]] const core::Geometry& geometry() const noexcept;

  /**
   * @brief Output enable
   */
  bool& output_enable() noexcept;

  /**
   * @brief Output enable
   */
  [[nodiscard]] bool output_enable() const noexcept;

  /**
   * @brief Silent mode
   */
  bool& silent_mode() noexcept;

  /**
   * @brief Silent mode
   */
  [[nodiscard]] bool silent_mode() const noexcept;

  /**
   * @brief If true, the devices return FPGA info in all frames. The FPGA info can be read by fpga_info().
   */
  bool& reads_fpga_info() noexcept;

  /**
   * @brief If true, the devices return FPGA info in all frames. The FPGA info can be read by fpga_info().
   */
  [[nodiscard]] bool reads_fpga_info() const noexcept;

  /**
   * @brief If true, the fan will be forced to start.
   */
  bool& force_fan() noexcept;

  /**
   * @brief If true, the fan will be forced to start.
   */
  [[nodiscard]] bool force_fan() const noexcept;

  /**
   * @brief If true, the applied voltage to transducers is dropped to GND while transducers are not being outputting.
   */
  bool& output_balance() noexcept;

  /**
   * @brief If true, the applied voltage to transducers is dropped to GND while transducers are not being outputting.
   */
  [[nodiscard]] bool output_balance() const noexcept;

  /**
   * @brief If true, this controller check ack from devices.
   */
  bool& check_ack() noexcept;

  /**
   * @brief If true, this controller check ack from devices.
   */
  [[nodiscard]] bool check_ack() const noexcept;

  /**
   * @brief FPGA info
   *  \return ok with FPGA information if succeeded, or err with error message if failed
   *  \details the first bit of FPGA info represents whether the fan is running.
   */
  const std::vector<uint8_t>& fpga_info();

  /**
   * @brief Update control flag
   * \return if this function returns true and check_ack is true, it guarantees that the devices have processed the data.
   */
  bool update_ctrl_flag();

  /**
   * @brief Open device with a link.
   * @param[in] link Link
   */
  void open(core::LinkPtr link);

  /**
   * @brief Clear all data in hardware
   * \return if this function returns true and check_ack is true, it guarantees that the devices have processed the data.
   */
  bool clear();

  /**
   * @brief Close the controller
   * \return if this function returns true and check_ack is true, it guarantees that the devices have processed the data.
   */
  bool close();

  /**
   * @brief Stop outputting
   * \return if this function returns true and check_ack is true, it guarantees that the devices have processed the data.
   */
  bool stop();

  /**
   * @brief Pause outputting
   * \return if this function returns true and check_ack is true, it guarantees that the devices have processed the data.
   */
  bool pause();

  /**
   * @brief Resume outputting
   * \return if this function returns true and check_ack is true, it guarantees that the devices have processed the data.
   */
  bool resume();

  bool send(core::IDatagramHeader& header);

  bool send(core::IDatagramBody& body);

  bool send(core::IDatagramHeader& header, core::IDatagramBody& body);

  bool send(core::IDatagramBody& body, core::IDatagramHeader& header);

  /**
   * @brief Enumerate firmware information
   * \return firmware information list
   */
  [[nodiscard]] std::vector<FirmwareInfo> firmware_info_list();

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
    void add_gain(core::Gain& gain) const;

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
  class STMTimerCallback final : core::CallbackHandler {
   public:
    friend class STMController;

    virtual ~STMTimerCallback() = default;
    STMTimerCallback(const STMTimerCallback& v) noexcept = delete;
    STMTimerCallback& operator=(const STMTimerCallback& obj) = delete;
    STMTimerCallback(STMTimerCallback&& obj) = delete;
    STMTimerCallback& operator=(STMTimerCallback&& obj) = delete;

    explicit STMTimerCallback(core::LinkPtr link) : _link(std::move(link)), _idx(0), _lock(false) {}

    void add(core::TxDatagram tx);
    void clear();
    void callback() override;

   private:
    core::LinkPtr _link;
    std::vector<core::TxDatagram> _txs;
    size_t _idx;
    std::atomic<bool> _lock;
  };

  [[nodiscard]] bool wait_msg_processed(uint8_t msg_id, size_t max_trial = 50);

  core::LinkPtr _link;
  core::Geometry _geometry;
  ControllerProps _props;
  bool _check_ack;
  core::TxDatagram _tx_buf;
  core::RxDatagram _rx_buf;

  std::vector<uint8_t> _fpga_infos;
};
}  // namespace autd
