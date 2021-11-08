// File: lpf.hpp
// Project: filter
// Created Date: 08/11/2021
// Author: Shun Suzuki
// -----
// Last Modified: 08/11/2021
// Modified By: Shun Suzuki (suzuki@hapis.k.u-tokyo.ac.jp)
// -----
// Copyright (c) 2021 Hapis Lab. All rights reserved.
//

#pragma once

#include <vector>

#include "autd3/core/modulation.hpp"

namespace autd::modulation::filter {

class FIR : public core::Filter {
 public:
  FIR(const uint16_t sampling_freq_div, const std::vector<double>& coef) noexcept : _sampling_freq_div(sampling_freq_div), _coef(coef) {}
  virtual ~FIR() = default;
  FIR(const FIR& v) noexcept = default;
  FIR& operator=(const FIR& obj) = default;
  FIR(FIR&& obj) = default;
  FIR& operator=(FIR&& obj) = default;

  void apply(const core::ModulationPtr& mod) const override;

  /**
   * @brief Low-pass filter for silent mode
   * @return FIR Filter which is equivelent for FPGA implementation
   */
  static FIR lpf();

 private:
  uint16_t _sampling_freq_div;
  std::vector<double> _coef;
};

}  // namespace autd::modulation::filter
