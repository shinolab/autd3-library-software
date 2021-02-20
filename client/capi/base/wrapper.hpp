// File: wrapper.hpp
// Project: capi
// Created Date: 09/06/2020
// Author: Shun Suzuki
// -----
// Last Modified: 20/02/2021
// Modified By: Shun Suzuki (suzuki@hapis.k.u-tokyo.ac.jp)
// -----
// Copyright (c) 2020 Hapis Lab. All rights reserved.
//

#pragma once

#include <utility>
#include <vector>

#include "controller.hpp"
#include "firmware_version.hpp"
#include "modulation.hpp"
#include "sequence.hpp"

typedef struct {
  autd::ModulationPtr ptr;
} ModulationWrapper;

typedef struct {
  autd::SequencePtr ptr;
} SequenceWrapper;

typedef struct {
  autd::ControllerPtr ptr;
} ControllerWrapper;

typedef struct {
  std::vector<autd::FirmwareInfo> list;
} FirmwareInfoListWrapper;

inline ModulationWrapper* ModulationCreate(const autd::ModulationPtr& ptr) { return new ModulationWrapper{ptr}; }
inline void ModulationDelete(ModulationWrapper* ptr) { delete ptr; }

inline SequenceWrapper* SequencePtrCreate(const autd::SequencePtr& ptr) { return new SequenceWrapper{ptr}; }
inline void SequenceDelete(SequenceWrapper* ptr) { delete ptr; }

inline ControllerWrapper* ControllerCreate(autd::ControllerPtr ptr) { return new ControllerWrapper{std::move(ptr)}; }
inline void ControllerDelete(ControllerWrapper* ptr) { delete ptr; }

inline FirmwareInfoListWrapper* FirmwareInfoListCreate(const std::vector<autd::FirmwareInfo>& list) { return new FirmwareInfoListWrapper{list}; }
inline void FirmwareInfoListDelete(FirmwareInfoListWrapper* ptr) { delete ptr; }
