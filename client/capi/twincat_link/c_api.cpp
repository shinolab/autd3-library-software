// File: c_api.cpp
// Project: capi
// Created Date: 02/07/2018
// Author: Shun Suzuki
// -----
// Last Modified: 20/02/2021
// Modified By: Shun Suzuki (suzuki@hapis.k.u-tokyo.ac.jp)
// -----
// Copyright (c) 2018-2020 Hapis Lab. All rights reserved.
//

#include <cstdint>

#include "../base/wrapper_link.hpp"
#include "./autd3_c_api_twincat_link.h"
#include "twincat_link.hpp"

void AUTDTwinCATLink(VOID_PTR* out, const char* ipv4_addr, const char* ams_net_id) {
  auto* link = LinkCreate(autd::link::TwinCATLink::Create(std::string(ipv4_addr), std::string(ams_net_id)));
  *out = link;
}
void AUTDLocalTwinCATLink(VOID_PTR* out) {
  auto* link = LinkCreate(autd::link::LocalTwinCATLink::Create());
  *out = link;
}