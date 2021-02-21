// File: gain.cpp
// Project: lib
// Created Date: 01/06/2016
// Author: Seki Inoue
// -----
// Last Modified: 20/02/2021
// Modified By: Shun Suzuki (suzuki@hapis.k.u-tokyo.ac.jp)
// -----
// Copyright (c) 2020 Hapis Lab. All rights reserved.
//

#include "gain.hpp"

#include <memory>
#include <vector>

#include "consts.hpp"

namespace autd::gain {

GainPtr Gain::Create() { return std::make_shared<Gain>(); }

Gain::Gain() noexcept : _built(false), _geometry(nullptr) {}
Gain::Gain(std::vector<AUTDDataArray> data) noexcept : _built(false), _geometry(nullptr), _data(std::move(data)) {}

void Gain::Build() {
  if (this->built()) return;
  auto geometry = this->geometry();

  CheckAndInit(geometry, &this->_data);

  for (size_t i = 0; i < geometry->num_devices(); i++) this->_data[i].fill(0x0000);

  this->_built = true;
}

bool Gain::built() const noexcept { return this->_built; }

GeometryPtr Gain::geometry() const noexcept { return this->_geometry; }

void Gain::SetGeometry(const GeometryPtr& geometry) noexcept { this->_geometry = geometry; }

std::vector<AUTDDataArray>& Gain::data() { return this->_data; }

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
