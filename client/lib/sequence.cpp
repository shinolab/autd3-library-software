// File: sequence.cpp
// Project: lib
// Created Date: 01/07/2020
// Author: Shun Suzuki
// -----
// Last Modified: 22/12/2020
// Modified By: Shun Suzuki (suzuki@hapis.k.u-tokyo.ac.jp)
// -----
// Copyright (c) 2020 Hapis Lab. All rights reserved.
//

#include "sequence.hpp"

#define _USE_MATH_DEFINES
#include <math.h>

#include <algorithm>
#include <stdexcept>

#include "consts.hpp"
#include "vector3.hpp"

namespace autd::sequence {
PointSequence::PointSequence() noexcept {
  this->_sampl_freq_div = 1;
  this->_sent = 0;
}

SequencePtr PointSequence::Create() noexcept { return std::make_shared<PointSequence>(); }

SequencePtr PointSequence::Create(std::vector<Vector3> control_points) noexcept {
  auto ptr = std::make_shared<PointSequence>();
  ptr->_control_points = control_points;
  return ptr;
}

void PointSequence::AppendPoint(Vector3 point) {
  if (this->_control_points.size() + 1 > POINT_SEQ_BUFFER_SIZE_MAX)
    throw new std::runtime_error("Point sequence buffer overflow. Maximum available buffer size is " + std::to_string(POINT_SEQ_BUFFER_SIZE_MAX) +
                                 ".");

  this->_control_points.push_back(point);
}

void PointSequence::AppendPoints(std::vector<Vector3> points) {
  if (this->_control_points.size() + points.size() > POINT_SEQ_BUFFER_SIZE_MAX)
    throw new std::runtime_error("Point sequence buffer overflow. Maximum available buffer size is " + std::to_string(POINT_SEQ_BUFFER_SIZE_MAX) +
                                 ".");
  this->_control_points.reserve(this->_control_points.size() + points.size());
  this->_control_points.insert(std::end(this->_control_points), std::begin(points), std::end(points));
}

std::vector<Vector3> PointSequence::control_points() { return this->_control_points; }

double PointSequence::SetFrequency(double freq) {
  auto sample_freq = std::min(this->_control_points.size() * freq, POINT_SEQ_BASE_FREQ);
  auto div = static_cast<size_t>(POINT_SEQ_BASE_FREQ / sample_freq);
  auto lm_cycle = this->_control_points.size() * div;

  uint16_t actual_div;
  if (lm_cycle > POINT_SEQ_CLK_IDX_MAX) {
    actual_div = static_cast<uint16_t>(POINT_SEQ_CLK_IDX_MAX / this->_control_points.size());
  } else {
    actual_div = static_cast<uint16_t>(div);
  }

  this->_sampl_freq_div = actual_div;

  return this->frequency();
}

double PointSequence::frequency() { return this->sampling_frequency() / this->_control_points.size(); }

double PointSequence::sampling_frequency() { return POINT_SEQ_BASE_FREQ / this->_sampl_freq_div; }

size_t PointSequence::sent() { return _sent; }

uint16_t PointSequence::sampling_frequency_division() { return this->_sampl_freq_div; }

static Vector3 GetOrthogonal(Vector3 v) {
  auto a = Vector3::unit_x();
  if (v.angle(a) < M_PI / 2.0) {
    a = Vector3::unit_y();
  }
  return v.cross(a);
}

SequencePtr CircumSeq::Create(Vector3 center, Vector3 normal, double radius, size_t n) {
  normal = normal.normalized();
  auto n1 = GetOrthogonal(normal).normalized();
  auto n2 = normal.cross(n1).normalized();

  std::vector<Vector3> control_points;
  for (int i = 0; i < n; i++) {
    auto theta = 2.0 * M_PI / n * i;
    auto x = n1 * radius * cos(theta);
    auto y = n2 * radius * sin(theta);
    control_points.push_back(center + x + y);
  }
  return PointSequence::Create(control_points);
}
}  // namespace autd::sequence
