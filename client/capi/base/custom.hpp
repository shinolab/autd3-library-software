// File: custom.hpp
// Project: base
// Created Date: 03/11/2021
// Author: Shun Suzuki
// -----
// Last Modified: 09/12/2021
// Modified By: Shun Suzuki (suzuki@hapis.k.u-tokyo.ac.jp)
// -----
// Copyright (c) 2021 Hapis Lab. All rights reserved.
//

#pragma once

#include <algorithm>
#include <memory>
#include <utility>
#include <vector>

#include "autd3.hpp"

/**
 * @brief Gain that can set the phase and duty ratio freely
 */
class CustomGain final : public autd::core::Gain {
 public:
  /**
   * @param[in] data pointer to data of duty ratio and phase of each transducer
   * @param[in] data_length length of the data
   * @details The data length should be the same as the number of transducers you use. The data is 16 bit unsigned integer, where high 8bits
   * represents duty ratio and low 8bits represents phase
   */
  explicit CustomGain(const uint16_t* data, const size_t data_length) : Gain(), _raw_data(data_length) {
    std::memcpy(_raw_data.data(), data, data_length * sizeof(uint16_t));
  }

  void calc(const autd::core::Geometry& geometry) override { this->_data = std::move(this->_raw_data); }

  ~CustomGain() override = default;
  CustomGain(const CustomGain& v) noexcept = delete;
  CustomGain& operator=(const CustomGain& obj) = delete;
  CustomGain(CustomGain&& obj) = default;
  CustomGain& operator=(CustomGain&& obj) = default;

 private:
  std::vector<autd::core::Drive> _raw_data;
};

/**
 * @brief Custom wave modulation
 */
class CustomModulation final : public autd::modulation::Modulation {
 public:
  /**
   * @brief Generate function
   * @param[in] buffer data of modulation
   * @param freq_div_ratio sampling frequency division ratio
   */
  explicit CustomModulation(const std::vector<uint8_t>& buffer, const size_t freq_div_ratio = 10) : Modulation(freq_div_ratio) {
    this->_buffer = buffer;
  }

  void calc() override {}

  ~CustomModulation() override = default;
  CustomModulation(const CustomModulation& v) noexcept = delete;
  CustomModulation& operator=(const CustomModulation& obj) = delete;
  CustomModulation(CustomModulation&& obj) = default;
  CustomModulation& operator=(CustomModulation&& obj) = default;
};
