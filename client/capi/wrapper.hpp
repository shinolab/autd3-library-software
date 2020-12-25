// File: wrapper.hpp
// Project: capi
// Created Date: 09/06/2020
// Author: Shun Suzuki
// -----
// Last Modified: 25/12/2020
// Modified By: Shun Suzuki (suzuki@hapis.k.u-tokyo.ac.jp)
// -----
// Copyright (c) 2020 Hapis Lab. All rights reserved.
//

#pragma once

#include "autd3.hpp"

typedef struct {
  autd::GainPtr ptr;
} GainWrapper;

typedef struct {
  autd::ModulationPtr ptr;
} ModulationWrapper;

typedef struct {
  autd::SequencePtr ptr;
} SequenceWrapper;

typedef struct {
  autd::LinkPtr ptr;
} LinkWrapper;

typedef struct {
  autd::ControllerPtr ptr;
} ControllerWrapper;

typedef struct {
  autd::link::EtherCATAdapters adapters;
} EtherCATAdaptersWrapper;

typedef struct {
  autd::FirmwareInfoList list;
} FirmwareInfoListWrapper;

inline GainWrapper* GainCreate(const autd::GainPtr& ptr) { return new GainWrapper{ptr}; }
inline void GainDelete(GainWrapper* ptr) { delete ptr; }

inline ModulationWrapper* ModulationCreate(const autd::ModulationPtr& ptr) { return new ModulationWrapper{ptr}; }
inline void ModulationDelete(ModulationWrapper* ptr) { delete ptr; }

inline SequenceWrapper* SequencePtrCreate(const autd::SequencePtr& ptr) { return new SequenceWrapper{ptr}; }
inline void SequenceDelete(SequenceWrapper* ptr) { delete ptr; }

inline LinkWrapper* LinkCreate(autd::LinkPtr ptr) { return new LinkWrapper{std::move(ptr)}; }

inline void LinkDelete(LinkWrapper* ptr) { delete ptr; }

inline ControllerWrapper* ControllerCreate(const autd::ControllerPtr& ptr) { return new ControllerWrapper{ptr}; }
inline void ControllerDelete(ControllerWrapper* ptr) { delete ptr; }

inline EtherCATAdaptersWrapper* EtherCATAdaptersCreate(const autd::link::EtherCATAdapters& adapters) { return new EtherCATAdaptersWrapper{adapters}; }
inline void EtherCATAdaptersDelete(EtherCATAdaptersWrapper* ptr) { delete ptr; }

inline FirmwareInfoListWrapper* FirmwareInfoListCreate(const autd::FirmwareInfoList& list) { return new FirmwareInfoListWrapper{list}; }
inline void FirmwareInfoListDelete(FirmwareInfoListWrapper* ptr) { delete ptr; }
