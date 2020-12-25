// File: grouped_gain.cpp
// Project: lib
// Created Date: 07/09/2018
// Author: Shun Suzuki
// -----
// Last Modified: 25/12/2020
// Modified By: Shun Suzuki (suzuki@hapis.k.u-tokyo.ac.jp)
// -----
// Copyright (c) 2018-2020 Hapis Lab. All rights reserved.
//

#include <map>
#include <utility>

#include "consts.hpp"
#include "gain.hpp"

namespace autd::gain {

GainPtr GroupedGain::Create(const std::map<size_t, GainPtr>& gain_map) {
  GainPtr gain = std::make_shared<GroupedGain>(gain_map);
  return gain;
}

void GroupedGain::Build() {
  if (this->built()) return;

  auto geometry = this->geometry();

  CheckAndInit(geometry, &this->_data);

  for (const auto& [fst, snd] : this->_gain_map) {
    auto g = snd;
    g->SetGeometry(geometry);
    g->Build();
  }

  for (size_t i = 0; i < geometry->numDevices(); i++) {
    auto groupId = geometry->GroupIDForDeviceIdx(i);
    if (_gain_map.count(groupId)) {
      auto& data = _gain_map[groupId]->data();
      this->_data[i] = data[i];
    } else {
      this->_data[i] = std::vector<uint16_t>(NUM_TRANS_IN_UNIT, 0x0000);
    }
  }

  this->_built = true;
}
}  // namespace autd::gain
