// File: soem_link.cpp
// Project: lib
// Created Date: 24/08/2019
// Author: Shun Suzuki
// -----
// Last Modified: 18/02/2020
// Modified By: Shun Suzuki (suzuki@hapis.k.u-tokyo.ac.jp)
// -----
// Copyright (c) 2019-2020 Hapis Lab. All rights reserved.
//

#include "soem_link.hpp"

#include <algorithm>
#include <bitset>
#include <chrono>
#include <iostream>
#include <memory>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "libsoem.hpp"
#include "privdef.hpp"

namespace autd {

void internal::SOEMLink::Open(std::string ifname) {
  _cnt = std::make_unique<libsoem::SOEMController>();

  auto ifname_and_devNum = autd::split(ifname, ':');
  _devNum = stoi(ifname_and_devNum[1]);
  _ifname = ifname_and_devNum[0];
  _cnt->Open(_ifname.c_str(), _devNum, EC_SM3_CYCLE_TIME_NANO_SEC, EC_SYNC0_CYCLE_TIME_NANO_SEC, HEADER_SIZE, NUM_TRANS_IN_UNIT * 2,
             EC_INPUT_FRAME_SIZE);
  _isOpen = _cnt->isOpen();
}

void internal::SOEMLink::Close() {
  if (_isOpen) {
    _cnt->Close();
    _isOpen = false;
  }
}

void internal::SOEMLink::Send(size_t size, std::unique_ptr<uint8_t[]> buf) {
  if (_isOpen) {
    _cnt->Send(size, std::move(buf));
  }
}

bool internal::SOEMLink::isOpen() { return _isOpen; }

bool internal::SOEMLink::CalibrateModulation() {
  constexpr int SYNC0_STEP = EC_SYNC0_CYCLE_TIME_MICRO_SEC * MOD_SAMPLING_FREQ / (1000 * 1000);
  constexpr uint32_t MOD_PERIOD_MS = (uint32_t)((MOD_BUF_SIZE / MOD_SAMPLING_FREQ) * 1000);

  auto succeed_calib = [&](const std::vector<uint16_t> &v) {
    auto min = *std::min_element(v.begin(), v.end());
    for (size_t i = 0; i < v.size(); i++) {
      auto h = (v.at(i) & 0xC000) >> 14;
      auto base = v.at(i) & 0x3FFF;
      if (h != 1 || (base - min) % SYNC0_STEP != 0) return false;
    }

    return true;
  };

  auto success = false;
  std::vector<uint16_t> v;
  for (size_t i = 0; i < 10; i++) {
    _cnt->Close();
    _cnt->Open(_ifname.c_str(), _devNum, EC_SM3_CYCLE_TIME_NANO_SEC, MOD_PERIOD_MS * 1000 * 1000, HEADER_SIZE, NUM_TRANS_IN_UNIT * 2,
               EC_INPUT_FRAME_SIZE);
    auto size = sizeof(RxGlobalHeader);
    auto body = std::make_unique<uint8_t[]>(size);
    auto *header = reinterpret_cast<RxGlobalHeader *>(&body[0]);
    header->msg_id = 0xFF;
    Send(size, move(body));

    std::this_thread::sleep_for(std::chrono::milliseconds((_devNum / 5 + 2) * MOD_PERIOD_MS));

    _cnt->Close();
    _cnt->Open(_ifname.c_str(), _devNum, EC_SM3_CYCLE_TIME_NANO_SEC, EC_SYNC0_CYCLE_TIME_NANO_SEC, HEADER_SIZE, NUM_TRANS_IN_UNIT * 2,
               EC_INPUT_FRAME_SIZE);
    size = sizeof(RxGlobalHeader);
    body = std::make_unique<uint8_t[]>(size);
    header = reinterpret_cast<RxGlobalHeader *>(&body[0]);
    header->msg_id = 0xFF;
    Send(size, move(body));

    std::this_thread::sleep_for(std::chrono::milliseconds(300));
    v = _cnt->Read(EC_OUTPUT_FRAME_SIZE * _devNum);
    if (succeed_calib(v)) {
      success = true;
      break;
    }
  }

  if (!success) {
    std::cerr << "Failed to CalibrateModulation." << std::endl;
    std::cerr << "======= Modulation Log ========" << std::endl;
    for (size_t i = 0; i < v.size(); i++) {
      auto h = (v.at(i) & 0xC000) >> 14;
      auto base = v.at(i) & 0x3FFF;
      std::cerr << i << "," << static_cast<int>(h) << "," << static_cast<int>(base) << std::endl;
    }
    std::cerr << "===============================" << std::endl;
  }

  return success;
}
}  // namespace autd
