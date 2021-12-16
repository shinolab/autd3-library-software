// File: osal_callback.hpp
// Project: osal_timer
// Created Date: 01/06/2021
// Author: Shun Suzuki
// -----
// Last Modified: 16/12/2021
// Modified By: Shun Suzuki (suzuki@hapis.k.u-tokyo.ac.jp)
// -----
// Copyright (c) 2021 Hapis Lab. All rights reserved.
//

#pragma once

namespace autd::core::timer {
/**
 * \brief CallbackHandler handle callback function which is called by a Timer
 */
class CallbackHandler {
 public:
  virtual void callback() = 0;
};
}  // namespace autd::core::timer
