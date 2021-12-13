// File: c_api.cpp
// Project: soem_link
// Created Date: 08/03/2021
// Author: Shun Suzuki
// -----
// Last Modified: 13/12/2021
// Modified By: Shun Suzuki (suzuki@hapis.k.u-tokyo.ac.jp)
// -----
// Copyright (c) 2021 Hapis Lab. All rights reserved.
//

#include "../base/wrapper_link.hpp"
#include "./soem_link.h"
#include "autd3/link/soem.hpp"

typedef struct {
  std::vector<autd::link::EtherCATAdapter> adapters;
} EtherCATAdaptersWrapper;

inline EtherCATAdaptersWrapper* ether_cat_adapters_create(const std::vector<autd::link::EtherCATAdapter>& adapters) {
  return new EtherCATAdaptersWrapper{adapters};
}
inline void ether_cat_adapters_delete(const EtherCATAdaptersWrapper* ptr) { delete ptr; }

int32_t AUTDGetAdapterPointer(void** out) {
  const auto adapters = autd::link::SOEM::enumerate_adapters();
  *out = ether_cat_adapters_create(adapters);
  return static_cast<int32_t>(adapters.size());
}
void AUTDGetAdapter(void* p_adapter, const int32_t index, char* desc, char* name) {
  const auto* wrapper = static_cast<EtherCATAdaptersWrapper*>(p_adapter);
  const auto& desc_ = wrapper->adapters[index].desc;
  const auto& name_ = wrapper->adapters[index].name;
  std::char_traits<char>::copy(desc, desc_.c_str(), desc_.size() + 1);
  std::char_traits<char>::copy(name, name_.c_str(), name_.size() + 1);
}
void AUTDFreeAdapterPointer(void* p_adapter) {
  const auto* wrapper = static_cast<EtherCATAdaptersWrapper*>(p_adapter);
  ether_cat_adapters_delete(wrapper);
}

void AUTDLinkSOEM(void** out, const char* ifname, const int32_t device_num, const uint32_t cycle_ticks) {
  auto soem_link = autd::link::SOEM::create(std::string(ifname), static_cast<size_t>(device_num), cycle_ticks);
  auto* link = link_create(std::move(soem_link));
  *out = link;
}

void AUTDSetSOEMOnLost(void* link, void* callback) {
  const auto* link_ = static_cast<LinkWrapper*>(link);
  if (auto* soem_link = dynamic_cast<autd::link::SOEM*>(link_->ptr.get()); soem_link != nullptr)
    soem_link->on_lost([callback](const std::string& msg) { reinterpret_cast<OnLostCallback>(callback)(msg.c_str()); });
}
