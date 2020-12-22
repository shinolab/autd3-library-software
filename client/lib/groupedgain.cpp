// File: groupedgain.cpp
// Project: lib
// Created Date: 07/09/2018
// Author: Shun Suzuki
// -----
// Last Modified: 22/12/2020
// Modified By: Shun Suzuki (suzuki@hapis.k.u-tokyo.ac.jp)
// -----
// Copyright (c) 2018-2020 Hapis Lab. All rights reserved.
//

#include <stdio.h>

#include <map>

#include "consts.hpp"
#include "gain.hpp"
#include "privdef.hpp"

namespace autd::gain {

GainPtr GroupedGain::Create(std::map<int, GainPtr> gainmap) {
  auto gain = std::make_shared<GroupedGain>();
  gain->_gainmap = gainmap;
  gain->_geometry = nullptr;
  return gain;
}

void GroupedGain::Build() {
  if (this->built()) return;

  auto geometry = this->geometry();

  CheckAndInit(geometry, &this->_data);

  for (std::pair<int, GainPtr> p : this->_gainmap) {
    GainPtr g = p.second;
    g->SetGeometry(geometry);
    g->Build();
  }

  for (int i = 0; i < geometry->numDevices(); i++) {
    auto groupId = geometry->GroupIDForDeviceIdx(i);
    if (_gainmap.count(groupId)) {
      std::vector<std::vector<uint16_t>>& data = _gainmap[groupId]->data();
      this->_data[i] = data[i];
    } else {
      this->_data[i] = std::vector<uint16_t>(NUM_TRANS_IN_UNIT, 0x0000);
    }
  }

  this->_built = true;
}
}  // namespace autd::gain
