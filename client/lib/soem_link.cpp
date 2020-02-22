// File: soem_link.cpp
// Project: lib
// Created Date: 24/08/2019
// Author: Shun Suzuki
// -----
// Last Modified: 22/02/2020
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

#include "ec_config.hpp"
#include "libsoem.hpp"
#include "privdef.hpp"

namespace autd {

void internal::SOEMLink::Open(std::string ifname) {
  _cnt = libsoem::ISOEMController::Create();

  auto ifname_and_devNum = autd::split(ifname, ':');
  _dev_num = stoi(ifname_and_devNum[1]);
  _ifname = ifname_and_devNum[0];

  this->_config.ec_sm3_cyctime_ns = EC_SM3_CYCLE_TIME_NANO_SEC;
  this->_config.ec_sync0_cyctime_ns = EC_SYNC0_CYCLE_TIME_NANO_SEC;
  this->_config.header_size = HEADER_SIZE;
  this->_config.body_size = NUM_TRANS_IN_UNIT * 2;
  this->_config.input_frame_size = EC_INPUT_FRAME_SIZE;

  _cnt->Open(_ifname.c_str(), _dev_num, this->_config);
  _is_open = _cnt->is_open();
}

void internal::SOEMLink::Close() {
  if (_is_open) {
    _cnt->Close();
    _is_open = false;
  }
}

void internal::SOEMLink::Send(size_t size, std::unique_ptr<uint8_t[]> buf) {
  if (_is_open) {
    _cnt->Send(size, std::move(buf));
  }
}

bool internal::SOEMLink::is_open() { return _is_open; }

void internal::SOEMLink::SetWaitForProcessMsg(bool is_wait) { _cnt->SetWaitForProcessMsg(is_wait); }

bool internal::SOEMLink::CalibrateModulation() {
  struct CalibrationStatus {
    uint16_t header;
    uint16_t sync_base;
  };

  auto parse = [](const std::vector<uint16_t> &v) {
    std::vector<CalibrationStatus> statuses;
    for (size_t i = 0; i < v.size(); i++) {
      uint16_t h = v.at(i) & SYNC_HEADER_MASK;
      uint16_t base = v.at(i) & SYNC_BASE_MASK;
      statuses.push_back(CalibrationStatus{h, base});
    }
    return statuses;
  };

  auto succeed_calib = [](const std::vector<CalibrationStatus> &statuses) {
    uint16_t min_base = 0xFFFF;
    for (auto status : statuses) {
      if (status.header != SYNC_HEADER_SUCCES) return false;
      if (status.sync_base < min_base) min_base = status.sync_base;
    }

    for (auto status : statuses) {
      if ((status.sync_base - min_base) % SYNC0_STEP != 0) return false;
    }

    return true;
  };

  auto make_calib_body = [](size_t *size) {
    *size = sizeof(RxGlobalHeader);
    auto body = std::make_unique<uint8_t[]>(*size);
    auto *header = reinterpret_cast<RxGlobalHeader *>(&body[0]);
    header->msg_id = CALIBRATION_HEADER;
    return std::move(body);
  };

  auto success = false;
  std::vector<CalibrationStatus> statuses;

  libsoem::ECConfig calib_config = this->_config;
  calib_config.ec_sync0_cyctime_ns = MOD_PERIOD_MS * 1000 * 1000;

  for (size_t i = 0; i < 10; i++) {
    _cnt->Close();
    _cnt->Open(_ifname.c_str(), _dev_num, calib_config);
    size_t size;
    auto body = make_calib_body(&size);
    Send(size, std::move(body));

    std::this_thread::sleep_for(std::chrono::milliseconds((_dev_num / 5 + 2) * MOD_PERIOD_MS));

    _cnt->Close();
    _cnt->Open(_ifname.c_str(), _dev_num, this->_config);
    auto body2 = make_calib_body(&size);
    Send(size, std::move(body2));
    _cnt->WaitForProcessMsg(CALIBRATION_HEADER);

    auto v = _cnt->Read();
    statuses = parse(v);

    if (succeed_calib(statuses)) {
      success = true;
      break;
    }
  }

  if (!success) {
    std::cerr << "Failed to CalibrateModulation." << std::endl;
    std::cerr << "======= Modulation Log ========" << std::endl;
    for (size_t i = 0; i < statuses.size(); i++) {
      auto status = statuses[i];
      std::cerr << i << "," << static_cast<int>(status.header) << "," << static_cast<int>(status.sync_base) << std::endl;
    }
    std::cerr << "===============================" << std::endl;
  }

  return success;
}  // namespace autd
}  // namespace autd
