// File: controller_v_0_7.cpp
// Project: lib
// Created Date: 30/10/2020
// Author: Shun Suzuki
// -----
// Last Modified: 30/10/2020
// Modified By: Shun Suzuki (suzuki@hapis.k.u-tokyo.ac.jp)
// -----
// Copyright (c) 2020 Hapis Lab. All rights reserved.
//

#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <cstring>
#include <iostream>
#include <mutex>
#include <queue>
#include <string>
#include <thread>
#include <vector>

#include "controller.hpp"
#include "controller_impl.hpp"
#include "ec_config.hpp"
#include "emulator_link.hpp"
#include "firmware_version.hpp"
#include "geometry.hpp"
#include "link.hpp"
#include "privdef.hpp"
#include "sequence.hpp"
#include "timer.hpp"

namespace autd {

namespace _internal {

static inline uint16_t log2u(const uint32_t x) {
  unsigned long n;
#ifdef _MSC_VER
  _BitScanReverse(&n, x);
#else
  n = 31 - __builtin_clz(x)
#endif
  return static_cast<uint16_t>(n);
}

AUTDControllerV_0_7::AUTDControllerV_0_7() : AUTDControllerV_0_6() {}

AUTDControllerV_0_7::~AUTDControllerV_0_7() {}

bool AUTDControllerV_0_7::Calibrate(Configuration config) {
  this->_config = config;

  auto num_devices = this->_geometry->numDevices();
  auto size = sizeof(RxGlobalHeaderV_0_6) + sizeof(uint16_t) * NUM_TRANS_IN_UNIT * num_devices;
  auto body = std::make_unique<uint8_t[]>(size);

  auto *header = reinterpret_cast<RxGlobalHeaderV_0_6 *>(&body[0]);
  header->msg_id = CMD_INIT_REF_CLOCK;
  header->command = CMD_INIT_REF_CLOCK;

  auto mod_smpl_freq = static_cast<uint32_t>(_config.mod_sampling_freq());
  auto mod_buf_size = static_cast<uint32_t>(_config.mod_buf_size());

  if (mod_buf_size < mod_smpl_freq) {
    std::cerr << "Modulation buffer size must be not less than sampling frequency.\n";
    std::cerr << "Modulation buffer size is set to " << mod_smpl_freq << std::endl;
    this->_config.set_mod_buf_size(static_cast<MOD_BUF_SIZE>(mod_smpl_freq));
  }

  auto mod_idx_shift = log2u(MOD_SAMPLING_FREQ_BASE / mod_smpl_freq);
  auto ref_clk_cyc_shift = log2u(mod_buf_size / mod_smpl_freq);

  auto *cursor = reinterpret_cast<uint16_t *>(&body[0] + sizeof(RxGlobalHeaderV_0_6) / sizeof(body[0]));
  for (int i = 0; i < this->_geometry->numDevices(); i++) {
    cursor[i] = mod_idx_shift;
    cursor[i + 1] = ref_clk_cyc_shift;
    cursor += NUM_TRANS_IN_UNIT;
  }

  this->SendData(size, std::move(body));
  return this->WaitMsgProcessed(CMD_INIT_REF_CLOCK, 5000);
}
}  // namespace _internal
}  // namespace autd
