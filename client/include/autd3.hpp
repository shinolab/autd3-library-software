/*
 * File: autd3.hpp
 * Project: include
 * Created Date: 13/05/2016
 * Author: Seki Inoue
 * -----
 * Last Modified: 04/09/2019
 * Modified By: Shun Suzuki (suzuki@hapis.k.u-tokyo.ac.jp)
 * -----
 * Copyright (c) 2016-2019 Hapis Lab. All rights reserved.
 * 
 */

#ifndef autd3hpp_
#define autd3hpp_

#include <iostream>
#include <string>
#include <memory>
#if WIN32
#include <codeanalysis\warnings.h>
#pragma warning(push)
#pragma warning(disable \
                : ALL_CODE_ANALYSIS_WARNINGS)
#endif
#include <Eigen/Core>
#include <Eigen/Geometry>
#if WIN32
#pragma warning(pop)
#endif

#include "controller.hpp"
#include "geometry.hpp"
#include "gain.hpp"
#include "modulation.hpp"

#endif
