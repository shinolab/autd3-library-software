// File: sequence.cpp
// Project: lib
// Created Date: 01/07/2020
// Author: Shun Suzuki
// -----
// Last Modified: 21/02/2021
// Modified By: Shun Suzuki (suzuki@hapis.k.u-tokyo.ac.jp)
// -----
// Copyright (c) 2020 Hapis Lab. All rights reserved.
//

#include "sequence.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <string>
#include <utility>

#include "consts.hpp"

namespace autd::sequence {
PointSequence::PointSequence() noexcept : _sampling_freq_div(1), _sent(0) {}
PointSequence::PointSequence(std::vector<Vector3> control_points) noexcept
    : _control_points(std::move(control_points)), _sampling_freq_div(1), _sent(0) {}

SequencePtr PointSequence::Create() noexcept { return std::make_shared<PointSequence>(); }

SequencePtr PointSequence::Create(const std::vector<Vector3>& control_points) noexcept {
  auto ptr = std::make_shared<PointSequence>(control_points);
  return ptr;
}

void PointSequence::AppendPoint(const Vector3& point) {
  if (this->_control_points.size() + 1 > POINT_SEQ_BUFFER_SIZE_MAX) {
    std::cerr << "Point sequence buffer overflow. Maximum available buffer size is " + std::to_string(POINT_SEQ_BUFFER_SIZE_MAX) + ".\n";
    return;
  }

  this->_control_points.emplace_back(point);
}
void PointSequence::AppendPoints(const std::vector<Vector3>& points) {
  if (this->_control_points.size() + points.size() > POINT_SEQ_BUFFER_SIZE_MAX) {
    std::cerr << "Point sequence buffer overflow. Maximum available buffer size is " + std::to_string(POINT_SEQ_BUFFER_SIZE_MAX) + ".\n";
    return;
  }

  this->_control_points.reserve(this->_control_points.size() + points.size());
  for (const auto& p : points) {
    this->_control_points.emplace_back(p);
  }
}  // namespace autd::sequence

std::vector<Vector3> PointSequence::control_points() const { return this->_control_points; }

Float PointSequence::SetFrequency(const Float freq) {
  const auto sample_freq = std::min(static_cast<Float>(this->_control_points.size()) * freq, POINT_SEQ_BASE_FREQ);
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

Float PointSequence::frequency() const { return this->sampling_frequency() / static_cast<Float>(this->_control_points.size()); }

Float PointSequence::sampling_frequency() const { return POINT_SEQ_BASE_FREQ / static_cast<Float>(this->_sampling_freq_div); }

size_t& PointSequence::sent() { return _sent; }

uint16_t PointSequence::sampling_frequency_division() const { return this->_sampling_freq_div; }
}  // namespace autd::sequence
