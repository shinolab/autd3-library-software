// File: sequence.cpp
// Project: lib
// Created Date: 01/07/2020
// Author: Shun Suzuki
// -----
// Last Modified: 26/12/2020
// Modified By: Shun Suzuki (suzuki@hapis.k.u-tokyo.ac.jp)
// -----
// Copyright (c) 2020 Hapis Lab. All rights reserved.
//

#include "sequence.hpp"

#define _USE_MATH_DEFINES  // NOLINT
#include <math.h>

#include <algorithm>
#include <cmath>
#include <string>
#include <utility>

#include "consts.hpp"
#include "convert.hpp"
#include "vector3.hpp"

namespace autd::sequence {
PointSequence::PointSequence() noexcept : _sampling_freq_div(1), _sent(0) {}
PointSequence::PointSequence(std::vector<Vector3> control_points) noexcept
    : _control_points(std::move(control_points)), _sampling_freq_div(1), _sent(0) {}

SequencePtr PointSequence::Create() noexcept { return std::make_shared<PointSequence>(); }

SequencePtr PointSequence::Create(const std::vector<utils::Vector3>& control_points) noexcept {
  auto ptr = std::make_shared<PointSequence>(Convert(control_points));
  return ptr;
}

void PointSequence::AppendPoint(const utils::Vector3& point) {
  if (this->_control_points.size() + 1 > POINT_SEQ_BUFFER_SIZE_MAX) {
    std::cerr << "Point sequence buffer overflow. Maximum available buffer size is " + std::to_string(POINT_SEQ_BUFFER_SIZE_MAX) + ".\n";
    return;
  }

  this->_control_points.emplace_back(Convert(point));
}
void PointSequence::AppendPoints(const std::vector<utils::Vector3>& points) {
  if (this->_control_points.size() + points.size() > POINT_SEQ_BUFFER_SIZE_MAX) {
    std::cerr << "Point sequence buffer overflow. Maximum available buffer size is " + std::to_string(POINT_SEQ_BUFFER_SIZE_MAX) + ".\n";
    return;
  }

  this->_control_points.reserve(this->_control_points.size() + points.size());
  for (const auto& p : points) {
    this->_control_points.emplace_back(Convert(p));
  }
}  // namespace autd::sequence
#ifdef USE_EIGEN_AUTD
SequencePtr PointSequence::Create(const std::vector<Vector3>& control_points) noexcept {
  auto ptr = std::make_shared<PointSequence>(control_points);
  return ptr;
}
void PointSequence::AppendPoint(const Vector3& point) {
  if (this->_control_points.size() + 1 > POINT_SEQ_BUFFER_SIZE_MAX) {
    std::cerr << "Point sequence buffer overflow. Maximum available buffer size is " + std::to_string(POINT_SEQ_BUFFER_SIZE_MAX) + ".\n";
    return;
  }
  this->_control_points.push_back(point);
}
void PointSequence::AppendPoints(const std::vector<Vector3>& points) {
  if (this->_control_points.size() + points.size() > POINT_SEQ_BUFFER_SIZE_MAX) {
    std::cerr << "Point sequence buffer overflow. Maximum available buffer size is " + std::to_string(POINT_SEQ_BUFFER_SIZE_MAX) + ".\n";
    return;
  }
  this->_control_points.reserve(this->_control_points.size() + points.size());
  this->_control_points.insert(std::end(this->_control_points), std::begin(points), std::end(points));
}
#endif

std::vector<Vector3> PointSequence::control_points() const { return this->_control_points; }

double PointSequence::SetFrequency(const double freq) {
  const auto sample_freq = std::min(static_cast<double>(this->_control_points.size()) * freq, POINT_SEQ_BASE_FREQ);
  const auto div = static_cast<size_t>(POINT_SEQ_BASE_FREQ / sample_freq);
  const auto lm_cycle = this->_control_points.size() * div;

  uint16_t actual_div;
  if (lm_cycle > POINT_SEQ_CLK_IDX_MAX) {
    actual_div = static_cast<uint16_t>(POINT_SEQ_CLK_IDX_MAX / this->_control_points.size());
  } else {
    actual_div = static_cast<uint16_t>(div);
  }

  this->_sampling_freq_div = actual_div;

  return this->frequency();
}

double PointSequence::frequency() const { return this->sampling_frequency() / static_cast<double>(this->_control_points.size()); }

double PointSequence::sampling_frequency() const { return POINT_SEQ_BASE_FREQ / this->_sampling_freq_div; }

size_t& PointSequence::sent() { return _sent; }

uint16_t PointSequence::sampling_frequency_division() const { return this->_sampling_freq_div; }

static Vector3 GetOrthogonal(const Vector3& v) {
  const auto a = Vector3::UnitX();
  if (acos(v.dot(a)) < M_PI / 2.0) {
    const auto b = Vector3::UnitY();
    return v.cross(b);
  }

  return v.cross(a);
}

SequencePtr CreateImpl(const Vector3& center, const Vector3& normal, const double radius, const size_t n) {
  const auto normal_ = normal.normalized();
  const auto n1 = GetOrthogonal(normal_).normalized();
  const auto n2 = normal_.cross(n1).normalized();

  std::vector<Vector3> control_points;
  for (size_t i = 0; i < n; i++) {
    const auto theta = 2.0 * M_PI / static_cast<double>(n) * static_cast<double>(i);
    auto x = n1 * radius * cos(theta);
    auto y = n2 * radius * sin(theta);
    control_points.emplace_back(center + x + y);
  }
  return PointSequence::Create(control_points);
}

SequencePtr CircumSeq::Create(const utils::Vector3& center, const utils::Vector3& normal, const double radius, const size_t n) {
  return CreateImpl(Convert(center), Convert(normal), radius, n);
}

#ifdef USE_EIGEN_AUTD
SequencePtr CircumSeq::Create(const Vector3& center, const Vector3& normal, const double radius, const size_t n) {
  return CreateImpl(center, normal, radius, n);
}
#endif
}  // namespace autd::sequence
