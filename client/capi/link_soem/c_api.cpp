// File: c_api.cpp
// Project: link_soem
// Created Date: 08/03/2021
// Author: Shun Suzuki
// -----
// Last Modified: 11/05/2021
// Modified By: Shun Suzuki (suzuki@hapis.k.u-tokyo.ac.jp)
// -----
// Copyright (c) 2021 Hapis Lab. All rights reserved.
//

#include "../base/wrapper_link.hpp"
#include "./soem_link.h"
#include "link_soem.hpp"

typedef struct {
  autd::link::EtherCATAdapters adapters;
} EtherCATAdaptersWrapper;

inline EtherCATAdaptersWrapper* EtherCATAdaptersCreate(const autd::link::EtherCATAdapters& adapters) { return new EtherCATAdaptersWrapper{adapters}; }
inline void EtherCATAdaptersDelete(EtherCATAdaptersWrapper* ptr) { delete ptr; }

int32_t AUTDGetAdapterPointer(void** out) {
  size_t size;
  const auto adapters = autd::link::SOEMLink::EnumerateAdapters(&size);
  *out = EtherCATAdaptersCreate(adapters);
  return static_cast<int32_t>(size);
}
void AUTDGetAdapter(void* p_adapter, const int32_t index, char* desc, char* name) {
  auto* wrapper = static_cast<EtherCATAdaptersWrapper*>(p_adapter);
  const auto& desc_ = wrapper->adapters[index].first;
  const auto& name_ = wrapper->adapters[index].second;
  std::char_traits<char>::copy(desc, desc_.c_str(), desc_.size() + 1);
  std::char_traits<char>::copy(name, name_.c_str(), name_.size() + 1);
}
void AUTDFreeAdapterPointer(void* p_adapter) {
  auto* wrapper = static_cast<EtherCATAdaptersWrapper*>(p_adapter);
  EtherCATAdaptersDelete(wrapper);
}

void AUTDSOEMLink(void** out, const char* ifname, const int32_t device_num) {
  auto* link = LinkCreate(autd::link::SOEMLink::Create(std::string(ifname), device_num));
  *out = link;
}
