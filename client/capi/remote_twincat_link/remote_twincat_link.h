// File: autd3_c_api_twincat_link.h
// Project: twincat_link
// Created Date: 20/02/2021
// Author: Shun Suzuki
// -----
// Last Modified: 14/12/2021
// Modified By: Shun Suzuki (suzuki@hapis.k.u-tokyo.ac.jp)
// -----
// Copyright (c) 2021 Hapis Lab. All rights reserved.
//

#pragma once

#include "../base/header.h"

#ifdef __cplusplus
extern "C" {
#endif
EXPORT_AUTD void AUTDLinkRemoteTwinCAT(void** out, const char* remote_ip_addr, const char* remote_ams_net_id, const char* local_ams_net_id);
#ifdef __cplusplus
}
#endif
