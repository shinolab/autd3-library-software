// File: sequence.hpp
// Project: core
// Created Date: 14/05/2021
// Author: Shun Suzuki
// -----
// Last Modified: 20/05/2021
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

#include "geometry.hpp"
#include "result.hpp"

namespace autd::core {
class PointSequence;
using SequencePtr = std::shared_ptr<PointSequence>;

/**
 * @brief PointSequence provides a function to display the focus sequentially and periodically.
 * @details PointSequence uses a timer on the FPGA to ensure that the focus is precisely timed.
 * PointSequence currently has the following three limitations.
 * 1. The maximum number of control points is 40000.
 * 2. The sampling interval of Control Points is an integer multiple of 25us.
 * 3. Only a single focus can be displayed at a certain moment.
 */
class PointSequence {
 public:
  PointSequence() noexcept : _sampling_freq_div(1), _sent(0) {}
  explicit PointSequence(std::vector<Vector3> control_points) noexcept
      : _control_points(std::move(control_points)), _sampling_freq_div(1), _sent(0) {}

  /**
   * @brief Generate empty PointSequence.
   */
  static SequencePtr create() noexcept { return std::make_shared<PointSequence>(); }

  /**
   * @brief Generate PointSequence with control points.
   */
  static SequencePtr create(const std::vector<Vector3>& control_points) noexcept { return std::make_shared<PointSequence>(control_points); }

  /**
   * @brief Add control point
   * @param[in] point control point
   * \return ok(whether succeeded), or err(error message) if unrecoverable error is occurred
   */
  [[nodiscard]] Error add_point(const Vector3& point) {
    if (this->_control_points.size() + 1 > POINT_SEQ_BUFFER_SIZE_MAX)
      return Err(std::string("Point sequence buffer overflow. Maximum available buffer size is " + std::to_string(POINT_SEQ_BUFFER_SIZE_MAX)));

    this->_control_points.emplace_back(point);
    return Ok(true);
  }

  /**
   * @brief Add control points
   * @param[in] points control point
   * \return ok(whether succeeded), or err(error message) if unrecoverable error is occurred
   */
  [[nodiscard]] Error add_points(const std::vector<Vector3>& points) {
    if (this->_control_points.size() + points.size() > POINT_SEQ_BUFFER_SIZE_MAX)
      return Err(std::string("Point sequence buffer overflow. Maximum available buffer size is " + std::to_string(POINT_SEQ_BUFFER_SIZE_MAX)));

    this->_control_points.reserve(this->_control_points.size() + points.size());
    for (const auto& p : points) this->_control_points.emplace_back(p);

    return Ok(true);
  }

  /**
   * @return std::vector<Vector3> Control points of the sequence
   */
  [[nodiscard]] std::vector<Vector3>& control_points() { return this->_control_points; }

  /**
   * @brief Set frequency of the sequence
   * @param[in] freq Frequency of the sequence
   * @details The Point Sequence Mode has two constraints, which determine the actual frequency of the sequence.
   * 1. The maximum number of control points is 40000.
   * 2. The sampling interval of control points is an integer multiple of 25us and less than 25us x 65536.
   * @return double Actual frequency of sequence
   */
  double set_frequency(const double freq) {
    const auto sample_freq = static_cast<double>(this->_control_points.size()) * freq;
    this->_sampling_freq_div = static_cast<uint16_t>(static_cast<double>(POINT_SEQ_BASE_FREQ) / sample_freq);
    return this->frequency();
  }

  /**
   * @return frequency of sequence
   */
  [[nodiscard]] double frequency() const { return this->sampling_frequency() / static_cast<double>(this->_control_points.size()); }

  /**
   * @return period of sequence
   */
  [[nodiscard]] size_t period_us() const { return this->sampling_period_us() * this->_control_points.size(); }

  /**
   * The sampling period is limited to an integer multiple of 25us. Therefore, the sampling frequency must be 40kHz/N.
   * @return double Sampling frequency of sequence
   */
  [[nodiscard]] double sampling_frequency() const { return static_cast<double>(POINT_SEQ_BASE_FREQ) / static_cast<double>(this->_sampling_freq_div); }

  /**
   * @return sampling period of sequence in micro seconds
   */
  [[nodiscard]] size_t sampling_period_us() const { return static_cast<size_t>(this->_sampling_freq_div) * 1000000 / POINT_SEQ_BASE_FREQ; }

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
  GeometryPtr _geometry;
  std::vector<Vector3> _control_points;
  uint16_t _sampling_freq_div;
  size_t _sent;
};
}  // namespace autd::core
