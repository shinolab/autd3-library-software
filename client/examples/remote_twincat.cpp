// File: remote_twincat.cpp
// Project: examples
// Created Date: 30/09/2021
// Author: Shun Suzuki
// -----
// Last Modified: 09/12/2021
// Modified By: Shun Suzuki (suzuki@hapis.k.u-tokyo.ac.jp)
// -----
// Copyright (c) 2021 Hapis Lab. All rights reserved.
//

#include "autd3/link/remote_twincat.hpp"

#include "autd3.hpp"
#include "runner.hpp"

int main() try {
  autd::Controller autd;
  autd.geometry().add_device(autd::Vector3(0, 0, 0), autd::Vector3(0, 0, 0));

  const std::string remote_ipv4_addr = "your server ip";
  const std::string remote_ams_net_id = "your server ams net id";
  const std::string local_ams_net_id = "your client ams net id";
  autd.open(autd::link::RemoteTwinCAT::create(remote_ipv4_addr, remote_ams_net_id, local_ams_net_id));

  return run(std::move(autd));
} catch (std::exception& e) {
  std::cerr << e.what() << std::endl;
  return -1;
}