// File: sequence.hpp
// Project: include
// Created Date: 01/07/2020
// Author: Shun Suzuki
// -----
// Last Modified: 24/12/2020
// Modified By: Shun Suzuki (suzuki@hapis.k.u-tokyo.ac.jp)
// -----
// Copyright (c) 2020 Hapis Lab. All rights reserved.
//

#pragma once

#include <vector>

#include "core.hpp"
#include "geometry.hpp"

namespace autd {
namespace sequence {
/**
 * @brief PointSequence provides a function to display the focus sequentially and periodically.
 * @details PointSequence uses a timer on the FPGA to ensure that the focus is precisely timed.
 * PointSequence currently has the following four limitations.
 * 1. The maximum number of control points is 40000.
 * 2. the sampling interval of Control Points is an integer multiple of 25us.
 * 3. (number of Control Points) x (sampling interval) must be less than or equal 1 second.
 * 4. Only a single focus can be displayed at a certain moment.
 */
class PointSequence {
 public:
  PointSequence() noexcept;
  /**
   * @brief Generate empty PointSequence.
   */
  static SequencePtr Create() noexcept;
  /**
   * @brief Generate PointSequence with control points.
   */
  static SequencePtr Create(std::vector<Vector3> control_points) noexcept;

  /**
   * @brief Append control point
   * @param[in] point control point
   * @details The maximum number of control points is 2000.
   */
  void AppendPoint(Vector3 point);
  /**
   * @brief Append control points
   * @param[in] points control points
   * @details The maximum number of control points is 2000.
   */
  void AppendPoints(std::vector<Vector3> points);
  /**
   * @return std::vector<Vector3> Control points of the sequence
   */
  [[nodiscard]] std::vector<Vector3> control_points() const;

  /**
   * @brief Set frequency of the sequence
   * @param[in] freq Frequency of the sequence
   * @details The Point Sequence Mode has three constraints, which determine the actual frequency of the sequence.
   * 1. The maximum number of control points is 40000.
   * 2. the sampling interval of Control Points is an integer multiple of 25us.
   * 3. (number of Control Points) x (sampling interval) must be less than or equal 1 second.
   * @return double Actual frequency of sequence
   */
  double SetFrequency(double freq);
  /**
   * @return double Frequency of sequence
   */
  [[nodiscard]] double frequency() const;
  /**
   * The sampling period is limited to an integer multiple of 25us. Therefore, the sampling frequency is 40kHz/N.
   * @return double Sampling frequency of sequence
   */
  [[nodiscard]] double sampling_frequency() const;
  /**
   * The sampling frequency division means the "(sampling period)/25us".
   * @return double Sampling frequency division
   */
  [[nodiscard]] uint16_t sampling_frequency_division() const;

  size_t& sent();

 private:
  GeometryPtr _geometry;
  std::vector<Vector3> _control_points;
  uint16_t _sampling_freq_div;
  size_t _sent;
};

/**
 * @brief Utility to generate PointSequence on a circumference.
 */
class CircumSeq : PointSequence {
 public:
  /**
   * @brief Generate PointSequence with control points on a circumference.
   * @param[in] center Center of the circumference
   * @param[in] normal Normal vector of the circumference
   * @param[in] radius Radius of the circumference
   * @param[in] n Number of the control points
   */
  static SequencePtr Create(Vector3 center, Vector3 normal, double radius, size_t n);
};
}  // namespace sequence
}  // namespace autd
