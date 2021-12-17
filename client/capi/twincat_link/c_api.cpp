// File: c_api.cpp
// Project: twincat_link
// Created Date: 08/03/2021
// Author: Shun Suzuki
// -----
// Last Modified: 12/12/2021
// Modified By: Shun Suzuki (suzuki@hapis.k.u-tokyo.ac.jp)
// -----
// Copyright (c) 2021 Hapis Lab. All rights reserved.
//

#include "../base/wrapper_link.hpp"
#include "./twincat_link.h"
#include "autd3/link/twincat.hpp"

void AUTDLinkTwinCAT(void** out) {
  auto* link = link_create(autd::link::TwinCAT::create());
  *out = link;
}
