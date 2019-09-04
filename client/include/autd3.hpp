/*
 *  autd3.hpp
 *  autd3
 *
 *  Created by Seki Inoue on 5/13/16.
 *  Modified by Shun Suzuki on 04/11/2018.
 *  Copyright © 2016-2019 Hapis Lab. All rights reserved.
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
#pragma warning(disable:ALL_CODE_ANALYSIS_WARNINGS)
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
