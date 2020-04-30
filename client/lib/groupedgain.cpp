// File: groupedgain.cpp
// Project: lib
// Created Date: 07/09/2018
// Author: Shun Suzuki
// -----
// Last Modified: 30/04/2020
// Modified By: Shun Suzuki (suzuki@hapis.k.u-tokyo.ac.jp)
// -----
// Copyright (c) 2018-2020 Hapis Lab. All rights reserved.
//

#include <stdio.h>

#include <map>

#include "consts.hpp"
#include "gain.hpp"
#include "privdef.hpp"

namespace autd {
namespace gain {

GainPtr GroupedGain::Create(std::map<int, GainPtr> gainmap) {
  auto gain = CreateHelper<GroupedGain>();
  gain->_gainmap = gainmap;
  gain->_geometry = nullptr;
  return gain;
}

void GroupedGain::Build() {
  if (this->built()) return;
  auto geo = this->geometry();
  if (geo == nullptr) {
    throw std::runtime_error("Geometry is required to build Gain");
  }

  this->_data.clear();

  const auto ndevice = geo->numDevices();
  for (int i = 0; i < ndevice; i++) {
    this->_data[geo->deviceIdForDeviceIdx(i)].resize(NUM_TRANS_IN_UNIT);
  }

  for (std::pair<int, GainPtr> p : this->_gainmap) {
    auto g = p.second;
    g->SetGeometry(geo);
    g->Build();
  }

  for (int i = 0; i < ndevice; i++) {
    auto groupId = geo->GroupIDForDeviceID(i);
    if (_gainmap.count(groupId)) {
      auto data = _gainmap[groupId]->data();
      this->_data[i] = data[i];
    } else {
      this->_data[i] = std::vector<uint16_t>(NUM_TRANS_IN_UNIT, 0x0000);
    }
  }

  this->_built = true;
}
}  // namespace gain
}  // namespace autd
