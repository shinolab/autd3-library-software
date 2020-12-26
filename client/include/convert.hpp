// File: convert.hpp
// Project: include
// Created Date: 26/12/2020
// Author: Shun Suzuki
// -----
// Last Modified: 26/12/2020
// Modified By: Shun Suzuki (suzuki@hapis.k.u-tokyo.ac.jp)
// -----
// Copyright (c) 2020 Hapis Lab. All rights reserved.
//

#pragma once

#include <vector>

#include "core.hpp"

namespace autd {
#ifdef USE_EIGEN_AUTD
inline Vector3 Convert(const utils::Vector3 v) { return Vector3(v.x(), v.y(), v.z()); }
inline std::vector<Vector3> Convert(const std::vector<utils::Vector3>& vin) {
  std::vector<Vector3> vs;
  vs.resize(vin.size());
  for (auto& v : vin) vs.emplace_back(Vector3(v.x(), v.y(), v.z()));
  return vs;
}
#else
inline Vector3 Convert(utils::Vector3 v) { return v; }
inline std::vector<Vector3> Convert(const std::vector<utils::Vector3>& vin) { return vin; }
#endif
}  // namespace autd
