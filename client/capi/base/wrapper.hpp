// File: wrapper.hpp
// Project: base
// Created Date: 08/03/2021
// Author: Shun Suzuki
// -----
// Last Modified: 19/05/2021
// Modified By: Shun Suzuki (suzuki@hapis.k.u-tokyo.ac.jp)
// -----
// Copyright (c) 2021 Hapis Lab. All rights reserved.
//

#pragma once

#include <vector>

#include "core/firmware_version.hpp"
#include "core/gain.hpp"
#include "core/modulation.hpp"
#include "core/sequence.hpp"
#include "linalg_backend.hpp"

typedef struct {
  autd::core::GainPtr ptr;
} GainWrapper;

inline GainWrapper* GainCreate(const autd::core::GainPtr& ptr) { return new GainWrapper{ptr}; }
inline void GainDelete(GainWrapper* ptr) { delete ptr; }

typedef struct {
  autd::core::ModulationPtr ptr;
} ModulationWrapper;

inline ModulationWrapper* ModulationCreate(const autd::core::ModulationPtr& ptr) { return new ModulationWrapper{ptr}; }
inline void ModulationDelete(ModulationWrapper* ptr) { delete ptr; }

typedef struct {
  autd::core::SequencePtr ptr;
} SequenceWrapper;

inline SequenceWrapper* SequenceCreate(const autd::core::SequencePtr& ptr) { return new SequenceWrapper{ptr}; }
inline void SequenceDelete(SequenceWrapper* ptr) { delete ptr; }

typedef struct {
  std::vector<autd::FirmwareInfo> list;
} FirmwareInfoListWrapper;

inline FirmwareInfoListWrapper* FirmwareInfoListCreate(const std::vector<autd::FirmwareInfo>& list) { return new FirmwareInfoListWrapper{list}; }
inline void FirmwareInfoListDelete(FirmwareInfoListWrapper* ptr) { delete ptr; }

typedef struct {
  autd::gain::holo::BackendPtr ptr;
} BackendWrapper;

inline BackendWrapper* BackendCreate(const autd::gain::holo::BackendPtr& ptr) { return new BackendWrapper{ptr}; }
inline void BackendDelete(BackendWrapper* ptr) { delete ptr; }
