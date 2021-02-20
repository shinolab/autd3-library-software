// File: matlab_gain.hpp
// Project: include
// Created Date: 20/02/2021
// Author: Shun Suzuki
// -----
// Last Modified: 20/02/2021
// Modified By: Shun Suzuki (suzuki@hapis.k.u-tokyo.ac.jp)
// -----
// Copyright (c) 2021 Hapis Lab. All rights reserved.
//

#pragma once

#include <string>
#include <utility>

#include "gain.hpp"

namespace autd {
namespace gain {

/**
 * @brief Gain created from Matlab mat file
 */
class MatlabGain final : public Gain {
 public:
  /**
   * @brief Generate function
   * @param[in] filename mat file path
   * @param[in] var_name variable name in mat file
   */
  static GainPtr Create(const std::string& filename, const std::string& var_name);
  void Build() override;
  MatlabGain(std::string filename, std::string var_name) : Gain(), _filename(std::move(filename)), _var_name(std::move(var_name)) {}
  ~MatlabGain() override = default;
  MatlabGain(const MatlabGain& v) noexcept = default;
  MatlabGain& operator=(const MatlabGain& obj) = default;
  MatlabGain(MatlabGain&& obj) = default;
  MatlabGain& operator=(MatlabGain&& obj) = default;

 protected:
  std::string _filename, _var_name;
};
}  // namespace gain
}  // namespace autd
