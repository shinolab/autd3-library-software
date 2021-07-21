﻿// File: c_api.cpp
// Project: soem_link
// Created Date: 08/03/2021
// Author: Shun Suzuki
// -----
// Last Modified: 21/07/2021
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

inline EtherCATAdaptersWrapper* EtherCATAdaptersCreate(const std::vector<autd::link::EtherCATAdapter>& adapters) {
  return new EtherCATAdaptersWrapper{adapters};
}
inline void EtherCATAdaptersDelete(EtherCATAdaptersWrapper* ptr) { delete ptr; }

int32_t AUTDGetAdapterPointer(void** out) {
  const auto adapters = autd::link::SOEM::enumerate_adapters();
  *out = EtherCATAdaptersCreate(adapters);
  return static_cast<int32_t>(adapters.size());
}
void AUTDGetAdapter(void* p_adapter, const int32_t index, char* desc, char* name) {
  auto* wrapper = static_cast<EtherCATAdaptersWrapper*>(p_adapter);
  const auto& desc_ = wrapper->adapters[index].desc;
  const auto& name_ = wrapper->adapters[index].name;
  std::char_traits<char>::copy(desc, desc_.c_str(), desc_.size() + 1);
  std::char_traits<char>::copy(name, name_.c_str(), name_.size() + 1);
}
void AUTDFreeAdapterPointer(void* p_adapter) {
  auto* wrapper = static_cast<EtherCATAdaptersWrapper*>(p_adapter);
  EtherCATAdaptersDelete(wrapper);
}

void AUTDLinkSOEM(void** out, const char* ifname, const int32_t device_num, const uint32_t cycle_ticks, ErrorHandler handler) {
  auto soem_link = autd::link::SOEM::create(std::string(ifname), static_cast<size_t>(device_num), cycle_ticks);
  if (handler != nullptr) soem_link->set_lost_handler([handler](const std::string& msg) { handler(msg.c_str()); });
  auto* link = LinkCreate(std::move(soem_link));
  *out = link;
}
