// File: c_api.cpp
// Project: capi
// Created Date: 02/07/2018
// Author: Shun Suzuki
// -----
// Last Modified: 20/02/2021
// Modified By: Shun Suzuki (suzuki@hapis.k.u-tokyo.ac.jp)
// -----
// Copyright (c) 2018-2020 Hapis Lab. All rights reserved.
//

#include "../base/wrapper_link.hpp"
#include "./soem_link.h"
#include "link/soem.hpp"

typedef struct {
  autd::link::EtherCATAdapters adapters;
} EtherCATAdaptersWrapper;

inline EtherCATAdaptersWrapper* EtherCATAdaptersCreate(const autd::link::EtherCATAdapters& adapters) { return new EtherCATAdaptersWrapper{adapters}; }
inline void EtherCATAdaptersDelete(EtherCATAdaptersWrapper* ptr) { delete ptr; }

int32_t AUTDGetAdapterPointer(VOID_PTR* out) {
  size_t size;
  const auto adapters = autd::link::SOEMLink::EnumerateAdapters(&size);
  *out = EtherCATAdaptersCreate(adapters);
  return static_cast<int32_t>(size);
}
void AUTDGetAdapter(VOID_PTR p_adapter, const int32_t index, char* desc, char* name) {
  auto* wrapper = static_cast<EtherCATAdaptersWrapper*>(p_adapter);
  const auto& desc_ = wrapper->adapters[index].first;
  const auto& name_ = wrapper->adapters[index].second;
  std::char_traits<char>::copy(desc, desc_.c_str(), desc_.size() + 1);
  std::char_traits<char>::copy(name, name_.c_str(), name_.size() + 1);
}
void AUTDFreeAdapterPointer(VOID_PTR p_adapter) {
  auto* wrapper = static_cast<EtherCATAdaptersWrapper*>(p_adapter);
  EtherCATAdaptersDelete(wrapper);
}

void AUTDSOEMLink(VOID_PTR* out, const char* ifname, const int32_t device_num) {
  auto* link = LinkCreate(autd::link::SOEMLink::Create(std::string(ifname), device_num));
  *out = link;
}
