// File: mod_from_file.hpp
// Project: examples
// Created Date: 05/07/2021
// Author: Shun Suzuki
// -----
// Last Modified: 09/12/2021
// Modified By: Shun Suzuki (suzuki@hapis.k.u-tokyo.ac.jp)
// -----
// Copyright (c) 2021 Hapis Lab. All rights reserved.
//

#pragma once

#include <filesystem>
#include <string>

#include "autd3.hpp"
#include "autd3/modulation/from_file.hpp"

namespace fs = std::filesystem;
using autd::NUM_TRANS_X, autd::NUM_TRANS_Y, autd::TRANS_SPACING_MM;

inline void mod_from_file_test(autd::Controller& autd) {
  autd.silent_mode() = true;

  const fs::path path = fs::path(std::string(AUTD3_RESOURCE_PATH)).append(std::string("sin150.wav"));
  const auto m = autd::modulation::Wav::create(path.string());

  const autd::Vector3 center(TRANS_SPACING_MM * ((NUM_TRANS_X - 1) / 2.0), TRANS_SPACING_MM * ((NUM_TRANS_Y - 1) / 2.0), 150.0);
  autd::gain::FocalPoint g(center);

  autd.send(g, m);
}
