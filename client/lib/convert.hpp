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

#if WIN32
#include <codeanalysis/warnings.h>  // NOLINT
#pragma warning(push)
#pragma warning(disable : ALL_CODE_ANALYSIS_WARNINGS)
#endif
#include <Eigen/Eigen>
#if WIN32
#pragma warning(pop)
#endif

#include <vector>

#include "core.hpp"
#include "quaternion.hpp"
#include "vector3.hpp"

namespace autd {
#ifdef USE_EIGEN_AUTD
inline const Vector3& Convert(const Vector3& v) { return v; }
inline Vector3 Convert(const utils::Vector3& v) { return Vector3(v.x(), v.y(), v.z()); }
inline std::vector<Vector3> Convert(const std::vector<utils::Vector3>& vin) {
  std::vector<Vector3> vs;
  vs.reserve(vin.size());
  for (const auto& v : vin) vs.emplace_back(Vector3(v.x(), v.y(), v.z()));
  return vs;
}
inline const Vector3& ConvertToEigen(const Vector3& v) { return v; }
inline Vector3 ConvertToEigen(const utils::Vector3& v) { return Vector3(v.x(), v.y(), v.z()); }
inline Quaternion ConvertToEigen(const utils::Quaternion& q) { return Quaternion(q.w(), q.x(), q.y(), q.z()); }
#else
#if WIN32
#include <codeanalysis/warnings.h>  // NOLINT
#pragma warning(push)
#pragma warning(disable : ALL_CODE_ANALYSIS_WARNINGS)
#endif
#include <Eigen/Eigen>
#if WIN32
#pragma warning(pop)
#endif
inline const Vector3& Convert(const Vector3& v) { return v; }
inline Vector3 Convert(Eigen::Matrix<Float, 3, 1> v) { return Vector3(v.x(), v.y(), v.z()); }
inline std::vector<Vector3> Convert(const std::vector<Vector3>& vin) { return vin; }
inline Eigen::Matrix<Float, 3, 1> ConvertToEigen(const Vector3& v) { return Eigen::Matrix<Float, 3, 1>(v.x(), v.y(), v.z()); }
inline Eigen::Quaternion<Float> ConvertToEigen(const Quaternion& q) { return Eigen::Quaternion<Float>(q.w(), q.x(), q.y(), q.z()); }
#endif
}  // namespace autd
