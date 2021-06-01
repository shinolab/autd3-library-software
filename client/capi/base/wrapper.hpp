// File: wrapper.hpp
// Project: base
// Created Date: 08/03/2021
// Author: Shun Suzuki
// -----
// Last Modified: 01/06/2021
// Modified By: Shun Suzuki (suzuki@hapis.k.u-tokyo.ac.jp)
// -----
// Copyright (c) 2021 Hapis Lab. All rights reserved.
//

#pragma once

#include <vector>

#include "controller.hpp"
#include "core/firmware_version.hpp"
#include "core/gain.hpp"
#include "core/modulation.hpp"
#include "core/sequence.hpp"
#include "linalg_backend.hpp"

typedef struct {
  std::unique_ptr<autd::Controller> ptr;
} ControllerWrapper;

inline ControllerWrapper* ControllerCreate(std::unique_ptr<autd::Controller> ptr) { return new ControllerWrapper{std::move(ptr)}; }
inline void ControllerDelete(ControllerWrapper* ptr) { delete ptr; }

typedef struct {
  std::unique_ptr<autd::Controller::STMController> ptr;
} STMControllerWrapper;

inline STMControllerWrapper* STMControllerCreate(std::unique_ptr<autd::Controller::STMController> ptr) {
  return new STMControllerWrapper{std::move(ptr)};
}
inline void STMControllerDelete(STMControllerWrapper* ptr) { delete ptr; }

typedef struct {
  std::unique_ptr<autd::Controller::STMTimer> ptr;
} STMTimerWrapper;

inline STMTimerWrapper* STMTimerCreate(std::unique_ptr<autd::Controller::STMTimer> ptr) { return new STMTimerWrapper{std::move(ptr)}; }
inline void STMTimerDelete(STMTimerWrapper* ptr) { delete ptr; }

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
