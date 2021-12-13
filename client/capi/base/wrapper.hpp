// File: wrapper.hpp
// Project: base
// Created Date: 08/03/2021
// Author: Shun Suzuki
// -----
// Last Modified: 13/12/2021
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
  autd::Controller::STMController stm;
} STMControllerWrapper;

inline STMControllerWrapper* stm_controller_create(autd::Controller::STMController stm) { return new STMControllerWrapper{std::move(stm)}; }
inline void stm_controller_delete(const STMControllerWrapper* ptr) { delete ptr; }

typedef struct {
  std::vector<autd::FirmwareInfo> list;
} FirmwareInfoListWrapper;

inline FirmwareInfoListWrapper* firmware_info_list_create(const std::vector<autd::FirmwareInfo>& list) { return new FirmwareInfoListWrapper{list}; }
inline void firmware_info_list_delete(const FirmwareInfoListWrapper* ptr) { delete ptr; }

typedef struct {
  autd::gain::holo::BackendPtr ptr;
} BackendWrapper;

inline BackendWrapper* backend_create(const autd::gain::holo::BackendPtr& ptr) { return new BackendWrapper{ptr}; }
inline void backend_delete(const BackendWrapper* ptr) { delete ptr; }
