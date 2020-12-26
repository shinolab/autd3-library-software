// File: grouped_gain.cpp
// Project: lib
// Created Date: 07/09/2018
// Author: Shun Suzuki
// -----
// Last Modified: 26/12/2020
// Modified By: Shun Suzuki (suzuki@hapis.k.u-tokyo.ac.jp)
// -----
// Copyright (c) 2018-2020 Hapis Lab. All rights reserved.
//

#include <map>
#include <utility>

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

  for (const auto& [fst, g] : this->_gain_map) {
    g->SetGeometry(geometry);
    g->Build();
  }

  for (size_t i = 0; i < geometry->num_devices(); i++) {
    auto group_id = geometry->group_id_for_device_idx(i);
    if (_gain_map.count(group_id)) {
      auto& data = _gain_map[group_id]->data();
      this->_data[i] = data[i];
    } else {
      this->_data[i] = AUTDDataArray{0x0000};
    }
  }

  this->_built = true;
}
}  // namespace autd::gain
