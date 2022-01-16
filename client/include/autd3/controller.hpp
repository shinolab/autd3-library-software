// File: controller.hpp
// Project: include
// Created Date: 05/11/2020
// Author: Shun Suzuki
// -----
// Last Modified: 15/01/2022
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
#include "core/type_traits.hpp"

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
  const std::vector<core::FPGAInfo>& fpga_info();

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
   * @brief Set silent step
   */
  bool set_silent_step(uint8_t step);

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

  template <class T>
  std::enable_if_t<core::type_traits::is_header_v<T>, bool> send(T&& header) {
    return send_impl(core::type_traits::to_header(header));
  }

  template <class T>
  std::enable_if_t<core::type_traits::is_body_v<T>, bool> send(T&& body) {
    return send_impl(core::type_traits::to_body(body));
  }

  template <class H, class B>
  std::enable_if_t<std::conjunction_v<core::type_traits::is_header<H>, core::type_traits::is_body<B>>, bool> send(H&& header, B&& body) {
    return send_impl(core::type_traits::to_body(body), core::type_traits::to_header(header));
  }

  template <class H, class B>
  std::enable_if_t<std::conjunction_v<core::type_traits::is_header<H>, core::type_traits::is_body<B>>, bool> send(B&& body, H&& header) {
    return send_impl(core::type_traits::to_body(body), core::type_traits::to_header(header));
  }

  /**
   * @brief Enumerate firmware information
   * \return firmware information list
   */
  [[nodiscard]] std::vector<FirmwareInfo> firmware_info_list();

  /**
   * \brief return pointer to software spatio-temporal modulation controller.
   * \details Never use Controller before calling STMController::finish
   */
  [[nodiscard]] STMController stm();

  /**
   * \brief Software spatio-temporal modulation controller.
   */
  class STMController {
   public:
    friend class Controller;

    /**
     * @brief Add gain for STM
     */
    void add(core::Gain& gain) const;

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

    class StreamCommaInputSTM {
      friend class Controller;

     public:
      explicit StreamCommaInputSTM(STMController& cnt) : _cnt(cnt) {}
      ~StreamCommaInputSTM() = default;
      StreamCommaInputSTM(const StreamCommaInputSTM& v) noexcept = delete;
      StreamCommaInputSTM& operator=(const StreamCommaInputSTM& obj) = delete;
      StreamCommaInputSTM(StreamCommaInputSTM&& obj) = default;
      StreamCommaInputSTM& operator=(StreamCommaInputSTM&& obj) = delete;

      template <class T>
      std::enable_if_t<core::type_traits::is_gain_v<T>, StreamCommaInputSTM&> operator<<(T&& gain) {
        _cnt.add(to_gain(gain));
        return *this;
      }

      template <class T>
      std::enable_if_t<core::type_traits::is_gain_v<T>, StreamCommaInputSTM&> operator,(T&& gain) {
        _cnt.add(to_gain(gain));
        return *this;
      }

     private:
      STMController& _cnt;
    };

    template <class T>
    std::enable_if_t<core::type_traits::is_gain_v<T>, StreamCommaInputSTM> operator<<(T&& gain) {
      this->add(core::type_traits::to_gain(gain));
      return StreamCommaInputSTM{*this};
    }

    ~STMController() { this->finish(); }
    STMController(const STMController& v) noexcept = delete;
    STMController& operator=(const STMController& obj) = delete;
    STMController(STMController&& obj) = default;
    STMController& operator=(STMController&& obj) = default;

   private:
    STMController() : _p_cnt(nullptr), _handler(nullptr), _timer(nullptr) {}
    explicit STMController(Controller* p_cnt, std::unique_ptr<STMTimerCallback> handler)
        : _p_cnt(p_cnt), _handler(std::move(handler)), _timer(nullptr) {}

    Controller* _p_cnt;
    std::unique_ptr<STMTimerCallback> _handler;
    std::unique_ptr<core::timer::Timer<STMTimerCallback>> _timer;
  };

 private:
  class STMTimerCallback final : core::timer::CallbackHandler {
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

  bool send_impl(core::datagram::IDatagramHeader& header);

  bool send_impl(core::datagram::IDatagramBody& body);

  bool send_impl(core::datagram::IDatagramHeader& header, core::datagram::IDatagramBody& body);

  bool send_impl(core::datagram::IDatagramBody& body, core::datagram::IDatagramHeader& header);

  [[nodiscard]] bool wait_msg_processed(uint8_t msg_id, size_t max_trial = 50);

  core::LinkPtr _link;
  core::Geometry _geometry;
  ControllerProps _props;
  bool _check_ack;
  core::TxDatagram _tx_buf;
  core::RxDatagram _rx_buf;

  std::vector<core::FPGAInfo> _fpga_infos;

 public:
  class StreamCommaInputHeader {
    friend class Controller;

   public:
    explicit StreamCommaInputHeader(Controller& cnt, core::datagram::IDatagramHeader& header) : _cnt(cnt), _header(header), _sent(false) {}
    ~StreamCommaInputHeader() {
      if (!_sent) _cnt.send(_header);
    }
    StreamCommaInputHeader(const StreamCommaInputHeader& v) noexcept = delete;
    StreamCommaInputHeader& operator=(const StreamCommaInputHeader& obj) = delete;
    StreamCommaInputHeader(StreamCommaInputHeader&& obj) = default;
    StreamCommaInputHeader& operator=(StreamCommaInputHeader&& obj) = delete;

    template <class B>
    std::enable_if_t<core::type_traits::is_body_v<B>> operator,(B&& body) {
      _cnt.send(_header, body);
      _sent = true;
    }

    template <class B>
    std::enable_if_t<core::type_traits::is_body_v<B>> operator<<(B&& body) {
      _cnt.send(_header, body);
      _sent = true;
    }

   private:
    Controller& _cnt;
    core::datagram::IDatagramHeader& _header;
    bool _sent;
  };

  class StreamCommaInputBody {
    friend class Controller;

   public:
    explicit StreamCommaInputBody(Controller& cnt, core::datagram::IDatagramBody& body) : _cnt(cnt), _body(body), _sent(false) {}
    ~StreamCommaInputBody() {
      if (!_sent) _cnt.send(_body);
    }
    StreamCommaInputBody(const StreamCommaInputBody& v) noexcept = delete;
    StreamCommaInputBody& operator=(const StreamCommaInputBody& obj) = delete;
    StreamCommaInputBody(StreamCommaInputBody&& obj) = default;
    StreamCommaInputBody& operator=(StreamCommaInputBody&& obj) = delete;

    template <class H>
    std::enable_if_t<core::type_traits::is_header_v<H>> operator,(H&& header) {
      _cnt.send(header, _body);
      _sent = true;
    }

    template <class H>
    std::enable_if_t<core::type_traits::is_header_v<H>> operator<<(H&& header) {
      _cnt.send(header, _body);
      _sent = true;
    }

   private:
    Controller& _cnt;
    core::datagram::IDatagramBody& _body;
    bool _sent;
  };

  template <class T>
  std::enable_if_t<core::type_traits::is_header_v<T>, StreamCommaInputHeader> operator<<(T&& header) {
    return StreamCommaInputHeader{*this, core::type_traits::to_header(header)};
  }
  template <class T>
  std::enable_if_t<core::type_traits::is_body_v<T>, StreamCommaInputBody> operator<<(T&& body) {
    return StreamCommaInputBody{*this, core::type_traits::to_body(body)};
  }
};

}  // namespace autd
