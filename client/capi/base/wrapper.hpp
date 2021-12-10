// File: wrapper.hpp
// Project: base
// Created Date: 08/03/2021
// Author: Shun Suzuki
// -----
// Last Modified: 10/12/2021
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
#include "autd3/gain/backend.hpp"

typedef struct {
  std::unique_ptr<autd::Controller::STMController> ptr;
} STMControllerWrapper;

inline STMControllerWrapper* STMControllerCreate(std::unique_ptr<autd::Controller::STMController> ptr) {
  return new STMControllerWrapper{std::move(ptr)};
}
inline void STMControllerDelete(const STMControllerWrapper* ptr) { delete ptr; }

typedef struct {
  std::vector<autd::FirmwareInfo> list;
} FirmwareInfoListWrapper;

inline FirmwareInfoListWrapper* FirmwareInfoListCreate(const std::vector<autd::FirmwareInfo>& list) { return new FirmwareInfoListWrapper{list}; }
inline void FirmwareInfoListDelete(const FirmwareInfoListWrapper* ptr) { delete ptr; }

typedef struct {
  autd::gain::holo::BackendPtr ptr;
} BackendWrapper;

inline BackendWrapper* BackendCreate(const autd::gain::holo::BackendPtr& ptr) { return new BackendWrapper{ptr}; }
inline void BackendDelete(const BackendWrapper* ptr) { delete ptr; }
