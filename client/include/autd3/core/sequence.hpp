// File: sequence.hpp
// Project: core
// Created Date: 14/05/2021
// Author: Shun Suzuki
// -----
// Last Modified: 21/07/2021
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
  explicit PointSequence(std::vector<Vector3> control_points) noexcept : Sequence(), _control_points(std::move(control_points)) {}

  size_t size() const override { return this->_control_points.size(); }

  /**
   * @brief Generate empty PointSequence.
   */
  static PointSequencePtr create() noexcept { return std::make_shared<PointSequence>(); }

  /**
   * @brief Generate PointSequence with control points.
   */
  static PointSequencePtr create(const std::vector<Vector3>& control_points) noexcept { return std::make_shared<PointSequence>(control_points); }

  /**
   * @brief Add control point
   * @param[in] point control point
   */
  void add_point(const Vector3& point) {
    if (this->_control_points.size() + 1 > POINT_SEQ_BUFFER_SIZE_MAX)
      throw core::SequenceBuildError(
          std::string("Point sequence buffer overflow. Maximum available buffer size is " + std::to_string(POINT_SEQ_BUFFER_SIZE_MAX)));

    this->_control_points.emplace_back(point);
  }

  /**
   * @brief Add control points
   * @param[in] points control point
   */
  void add_points(const std::vector<Vector3>& points) {
    if (this->_control_points.size() + points.size() > POINT_SEQ_BUFFER_SIZE_MAX)
      throw core::SequenceBuildError(
          std::string("Point sequence buffer overflow. Maximum available buffer size is " + std::to_string(POINT_SEQ_BUFFER_SIZE_MAX)));

    this->_control_points.reserve(this->_control_points.size() + points.size());
    for (const auto& p : points) this->_control_points.emplace_back(p);
  }

  /**
   * @return std::vector<Vector3> Control points of the sequence
   */
  [[nodiscard]] std::vector<Vector3>& control_points() { return this->_control_points; }

 private:
  std::vector<Vector3> _control_points;
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
  GainSequence() noexcept : Sequence() {}
  explicit GainSequence(std::vector<GainPtr> gains) noexcept : Sequence(), _gains(std::move(gains)) {}

  size_t size() const override { return this->_gains.size(); }

  /**
   * @brief Generate empty GainSequence
   */
  static GainSequencePtr create() noexcept { return std::make_shared<GainSequence>(); }

  /**
   * @brief Generate PointSequence with control points.
   */
  static GainSequencePtr create(const std::vector<GainPtr>& gains) noexcept { return std::make_shared<GainSequence>(gains); }

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
  [[nodiscard]] std::vector<GainPtr>& gains() { return this->_gains; }

 private:
  std::vector<GainPtr> _gains;
};
}  // namespace autd::core
