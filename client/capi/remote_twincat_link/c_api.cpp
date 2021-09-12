// File: c_api.cpp
// Project: twincat_link
// Created Date: 08/03/2021
// Author: Shun Suzuki
// -----
// Last Modified: 13/09/2021
// Modified By: Shun Suzuki (suzuki@hapis.k.u-tokyo.ac.jp)
// -----
// Copyright (c) 2021 Hapis Lab. All rights reserved.
//

#include "../base/wrapper_link.hpp"
#include "./remote_twincat_link.h"
#include "autd3/link/remote_twincat.hpp"

void AUTDLinkRemoteTwinCAT(void** out, const char* remote_ip_addr, const char* remote_ams_net_id, const char* local_ams_net_id) {
  const auto remote_ip_addr_ = std::string(remote_ip_addr);
  const auto remote_ams_net_id_ = std::string(remote_ams_net_id);
  const auto local_ams_net_id_ = std::string(local_ams_net_id);

  auto* link = LinkCreate(autd::link::RemoteTwinCAT::create(remote_ip_addr_, remote_ams_net_id_, local_ams_net_id_));
  *out = link;
}
