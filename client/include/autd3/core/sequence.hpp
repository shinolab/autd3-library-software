// File: sequence.hpp
// Project: core
// Created Date: 14/05/2021
// Author: Shun Suzuki
// -----
// Last Modified: 03/09/2021
// Modified By: Shun Suzuki (suzuki@hapis.k.u-tokyo.ac.jp)
// -----
// Copyright (c) 2021 Hapis Lab. All rights reserved.
//

#pragma once

#include <algorithm>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "exception.hpp"
#include "gain.hpp"
#include "geometry.hpp"

namespace autd::core {
class PointSequence;
class GainSequence;
using PointSequencePtr = std::shared_ptr<PointSequence>;
using GainSequencePtr = std::shared_ptr<GainSequence>;

class Sequence {
 public:
  Sequence() : _sampling_freq_div(1), _sent(0) {}
  virtual ~Sequence() = default;

  virtual size_t size() const = 0;

  /**
   * @brief Set frequency of the sequence
   * @param[in] freq Frequency of the sequence
   * @details The Point Sequence Mode has two constraints, which determine the actual frequency of the sequence.
   * 1. The maximum number of control points is 65536.
   * 2. The sampling interval of control points is an integer multiple of 25us and less than 25us x 65536.
   * @return double Actual frequency of sequence
   */
  double set_frequency(const double freq) {
    const auto sample_freq = std::clamp(static_cast<double>(this->size()) * freq, 0.0, static_cast<double>(SEQ_BASE_FREQ));
    this->_sampling_freq_div = static_cast<uint16_t>(static_cast<double>(SEQ_BASE_FREQ) / sample_freq);
    return this->frequency();
  }

  /**
   * @return frequency of sequence
   */
  [[nodiscard]] double frequency() const { return this->sampling_frequency() / static_cast<double>(this->size()); }

  /**
   * @return period of sequence
   */
  [[nodiscard]] size_t period_us() const { return this->sampling_period_us() * this->size(); }

  /**
   * The sampling period is limited to an integer multiple of 25us. Therefore, the sampling frequency must be 40kHz/N.
   * @return double Sampling frequency of sequence
   */
  [[nodiscard]] double sampling_frequency() const { return static_cast<double>(SEQ_BASE_FREQ) / static_cast<double>(this->_sampling_freq_div); }

  /**
   * @return sampling period of sequence in micro seconds
   */
  [[nodiscard]] size_t sampling_period_us() const { return static_cast<size_t>(this->_sampling_freq_div) * 1000000 / SEQ_BASE_FREQ; }

  /**
   * The sampling frequency division means the (sampling period)/25us.
   * @return double Sampling frequency division
   */
  [[nodiscard]] uint16_t sampling_frequency_division() const { return this->_sampling_freq_div; }

  /**
   * \brief sent means data length already sent to devices.
   */
  size_t& sent() { return _sent; }

 private:
  uint16_t _sampling_freq_div;
  size_t _sent;
};

/**
 * @brief PointSequence provides a function to display the focus sequentially and periodically.
 * @details PointSequence uses a timer on the FPGA to ensure that the focus is precisely timed.
 * PointSequence currently has the following three limitations.
 * 1. The maximum number of control points is 65536.
 * 2. The sampling interval of control points is an integer multiple of 25us and less than 25us x 65536.
 * 3. Only a single focus can be displayed at a certain moment.
 */
class PointSequence : virtual public Sequence {
 public:
  PointSequence() noexcept : Sequence() {}

  size_t size() const override { return this->_control_points.size(); }

  /**
   * @brief Generate empty PointSequence.
   */
  static PointSequencePtr create() noexcept { return std::make_shared<PointSequence>(); }

  /**
   * @brief Add control point
   * @param[in] point control point
   * @param[in] duty duty ratio
   */
  void add_point(const Vector3& point, const uint8_t duty = 0xFF) {
    if (this->_control_points.size() + 1 > POINT_SEQ_BUFFER_SIZE_MAX)
      throw core::SequenceBuildError(
          std::string("Point sequence buffer overflow. Maximum available buffer size is " + std::to_string(POINT_SEQ_BUFFER_SIZE_MAX)));

    this->_control_points.emplace_back(point);
    this->_duties.emplace_back(duty);
  }

  /**
   * @brief Add control points
   * @param[in] points control points
   * @param[in] duties duty ratios
   * @details duties.resize(points.size(), 0xFF) will be called.
   */
  void add_points(const std::vector<Vector3>& points, std::vector<uint8_t>& duties) {
    if (this->_control_points.size() + points.size() > POINT_SEQ_BUFFER_SIZE_MAX)
      throw core::SequenceBuildError(
          std::string("Point sequence buffer overflow. Maximum available buffer size is " + std::to_string(POINT_SEQ_BUFFER_SIZE_MAX)));

    this->_control_points.reserve(this->_control_points.size() + points.size());
    this->_control_points.insert(this->_control_points.end(), points.begin(), points.end());

    duties.resize(points.size(), 0xFF);
    this->_duties.reserve(this->_duties.size() + duties.size());
    this->_duties.insert(this->_duties.end(), duties.begin(), duties.end());
  }

  /**
   * @param[in] index control point index
   * @return Vector3 Control point of the sequence
   */
  [[nodiscard]] Vector3& control_point(size_t index) { return this->_control_points[index]; }

  /**
   * @param[in] index control point index
   * @return uint8_t Duty ratio of the sequence
   */
  [[nodiscard]] uint8_t& duty(size_t index) { return this->_duties[index]; }

  /**
   * @return std::vector<Vector3> Control points of the sequence
   */
  [[nodiscard]] const std::vector<Vector3>& control_points() { return this->_control_points; }

  /**
   * @return std::vector<uint8_t> Duty ratios of the sequence
   */
  [[nodiscard]] const std::vector<uint8_t>& duties() { return this->_duties; }

 private:
  std::vector<Vector3> _control_points;
  std::vector<uint8_t> _duties;
};

/**
 * @brief GainSequence provides a function to display Gain sequentially and periodically.
 * @details GainSequence uses a timer on the FPGA to ensure that Gain is precisely timed.
 * GainSequence currently has the following three limitations.
 * 1. The maximum number of gains is 1024.
 * 2. The sampling interval of gains is an integer multiple of 25us and less than 25us x 65536.
 */
class GainSequence : virtual public Sequence {
 public:
  GainSequence() noexcept : Sequence(), _gain_mode(GAIN_MODE::DUTY_PHASE_FULL) {}
  explicit GainSequence(GAIN_MODE gain_mode) noexcept : Sequence(), _gain_mode(gain_mode) {}
  explicit GainSequence(std::vector<GainPtr> gains, GAIN_MODE gain_mode) noexcept : Sequence(), _gains(std::move(gains)), _gain_mode(gain_mode) {}

  size_t size() const override { return this->_gains.size(); }

  /**
   * @brief Generate empty GainSequence
   */
  static GainSequencePtr create(GAIN_MODE gain_mode = GAIN_MODE::DUTY_PHASE_FULL) noexcept { return std::make_shared<GainSequence>(gain_mode); }

  /**
   * @brief Generate PointSequence with control points.
   */
  static GainSequencePtr create(const std::vector<GainPtr>& gains, GAIN_MODE gain_mode = GAIN_MODE::DUTY_PHASE_FULL) noexcept {
    return std::make_shared<GainSequence>(gains, gain_mode);
  }

  /**
   * @brief Add gain
   * @param[in] gain gain
   */
  void add_gain(const GainPtr& gain) {
    if (this->_gains.size() + 1 > GAIN_SEQ_BUFFER_SIZE_MAX)
      throw core::SequenceBuildError(
          std::string("Gain sequence buffer overflow. Maximum available buffer size is " + std::to_string(GAIN_SEQ_BUFFER_SIZE_MAX)));

    this->_gains.emplace_back(gain);
  }

  /**
   * @brief Add gains
   * @param[in] gains vector of gain
   */
  void add_points(const std::vector<GainPtr>& gains) {
    if (this->_gains.size() + gains.size() > GAIN_SEQ_BUFFER_SIZE_MAX)
      throw core::SequenceBuildError(
          std::string("Gain sequence buffer overflow. Maximum available buffer size is " + std::to_string(GAIN_SEQ_BUFFER_SIZE_MAX)));

    this->_gains.reserve(this->_gains.size() + gains.size());
    for (const auto& p : gains) this->_gains.emplace_back(p);
  }

  /**
   * @return std::vector<GainPtr> Gain list of the sequence
   */
  [[nodiscard]] const std::vector<GainPtr>& gains() { return this->_gains; }

  /**
   * @return GAIN_MODE
   */
  GAIN_MODE& gain_mode() { return this->_gain_mode; }

 private:
  std::vector<GainPtr> _gains;
  GAIN_MODE _gain_mode;
};
}  // namespace autd::core
