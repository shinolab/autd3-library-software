// File: custom.hpp
// Project: base
// Created Date: 03/11/2021
// Author: Shun Suzuki
// -----
// Last Modified: 19/11/2021
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
   * @brief Generate function
   * @param[in] data pointer to data of duty ratio and phase of each transducer
   * @param[in] data_length length of the data
   * @details The data length should be the same as the number of transducers you use. The data is 16 bit unsigned integer, where high 8bits
   * represents duty ratio and low 8bits represents phase
   */
  static autd::GainPtr create(const uint16_t* data, const size_t data_length) {
    const auto dev_num = (data_length + autd::NUM_TRANS_IN_UNIT - 1) / autd::NUM_TRANS_IN_UNIT;
    std::vector<autd::DataArray> raw_data(dev_num);
    for (size_t i = 0; i < dev_num; i++) {
      const auto rem = std::clamp(data_length - i * autd::NUM_TRANS_IN_UNIT, size_t{0}, autd::NUM_TRANS_IN_UNIT);
      std::memcpy(&raw_data[i][0], data + i * autd::NUM_TRANS_IN_UNIT, rem * sizeof(uint16_t));
    }
    return std::make_shared<CustomGain>(raw_data);
  }
  void calc(const autd::core::GeometryPtr& geometry) override { this->_data = std::move(this->_raw_data); }
  explicit CustomGain(std::vector<autd::DataArray> data) : autd::core::Gain(), _raw_data(std::move(data)) {}
  ~CustomGain() override = default;
  CustomGain(const CustomGain& v) noexcept = default;
  CustomGain& operator=(const CustomGain& obj) = default;
  CustomGain(CustomGain&& obj) = default;
  CustomGain& operator=(CustomGain&& obj) = default;

 private:
  std::vector<autd::DataArray> _raw_data;
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
  static autd::ModulationPtr create(const std::vector<uint8_t>& buffer, const size_t freq_div_ratio = 10) {
    return std::make_shared<CustomModulation>(buffer, freq_div_ratio);
  }
  void calc() override {}
  explicit CustomModulation(const std::vector<uint8_t>& buffer, const size_t freq_div_ratio) : autd::core::Modulation(freq_div_ratio) {
    this->_buffer = buffer;
  }
};
