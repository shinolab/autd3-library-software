// File: autd3.hpp
// Project: include
// Created Date: 13/05/2016
// Author: Seki Inoue
// -----
// Last Modified: 18/02/2020
// Modified By: Shun Suzuki (suzuki@hapis.k.u-tokyo.ac.jp)
// -----
// Copyright (c) 2016-2020 Hapis Lab. All rights reserved.
//

#ifndef INCLUDE_AUTD3_HPP_
#define INCLUDE_AUTD3_HPP_

#include <iostream>
#include <memory>
#include <string>
#if WIN32
#include <codeanalysis\warnings.h>
#pragma warning(push)
#pragma warning(disable : ALL_CODE_ANALYSIS_WARNINGS)
#endif
#include <Eigen/Core>
#include <Eigen/Geometry>
#if WIN32
#pragma warning(pop)
#endif

#include "controller.hpp"
#include "gain.hpp"
#include "geometry.hpp"
#include "modulation.hpp"

#endif  // INCLUDE_AUTD3_HPP_
