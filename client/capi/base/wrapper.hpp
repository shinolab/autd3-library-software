// File: wrapper.hpp
// Project: base
// Created Date: 08/03/2021
// Author: Shun Suzuki
// -----
// Last Modified: 09/09/2021
// Modified By: Shun Suzuki (suzuki@hapis.k.u-tokyo.ac.jp)
// -----
// Copyright (c) 2021 Hapis Lab. All rights reserved.
//

#pragma once

#include <memory>
#include <utility>
#include <vector>

#include "autd3/controller.hpp"
#include "autd3/core/firmware_version.hpp"
#include "autd3/core/gain.hpp"
#include "autd3/core/sequence.hpp"

typedef struct {
  autd::ControllerPtr ptr;
} ControllerWrapper;

inline ControllerWrapper* ControllerCreate(autd::ControllerPtr ptr) { return new ControllerWrapper{std::move(ptr)}; }
inline void ControllerDelete(const ControllerWrapper* ptr) { delete ptr; }

typedef struct {
  std::unique_ptr<autd::Controller::STMController> ptr;
} STMControllerWrapper;

inline STMControllerWrapper* STMControllerCreate(std::unique_ptr<autd::Controller::STMController> ptr) {
  return new STMControllerWrapper{std::move(ptr)};
}
inline void STMControllerDelete(const STMControllerWrapper* ptr) { delete ptr; }

typedef struct {
  autd::core::GainPtr ptr;
} GainWrapper;

inline GainWrapper* GainCreate(const autd::core::GainPtr& ptr) { return new GainWrapper{ptr}; }
inline void GainDelete(const GainWrapper* ptr) { delete ptr; }

typedef struct {
  std::shared_ptr<autd::core::Sequence> ptr;
} SequenceWrapper;

inline SequenceWrapper* SequenceCreate(const std::shared_ptr<autd::core::Sequence>& ptr) { return new SequenceWrapper{ptr}; }
inline void SequenceDelete(const SequenceWrapper* ptr) { delete ptr; }

typedef struct {
  std::vector<autd::FirmwareInfo> list;
} FirmwareInfoListWrapper;

inline FirmwareInfoListWrapper* FirmwareInfoListCreate(const std::vector<autd::FirmwareInfo>& list) { return new FirmwareInfoListWrapper{list}; }
inline void FirmwareInfoListDelete(const FirmwareInfoListWrapper* ptr) { delete ptr; }
