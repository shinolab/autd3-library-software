// File: controller.cpp
// Project: lib
// Created Date: 13/05/2016
// Author: Seki Inoue
// -----
// Last Modified: 30/10/2020
// Modified By: Shun Suzuki (suzuki@hapis.k.u-tokyo.ac.jp)
// -----
// Copyright (c) 2016-2020 Hapis Lab. All rights reserved.
//

#include "controller.hpp"

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

ControllerPtr Controller::Create(AUTD_VERSION version) {
  switch (version) {
    case AUTD_VERSION::V_0_1:
      return std::make_shared<_internal::AUTDControllerV_0_1>();
    case AUTD_VERSION::V_0_6:
      return std::make_shared<_internal::AUTDControllerV_0_6>();
    case AUTD_VERSION::V_0_7:
      return std::make_shared<_internal::AUTDControllerV_0_7>();
  }
  return nullptr;
}

AUTDController::AUTDController() {
  this->_link = nullptr;
  this->_geometry = Geometry::Create();
  this->_silent_mode = true;
  this->_p_stm_timer = std::make_unique<Timer>();
}

AUTDController::~AUTDController() {
  if (std::this_thread::get_id() != this->_build_gain_thr.get_id() && this->_build_gain_thr.joinable()) this->_build_gain_thr.join();
  if (std::this_thread::get_id() != this->_build_mod_thr.get_id() && this->_build_mod_thr.joinable()) this->_build_mod_thr.join();
  if (std::this_thread::get_id() != this->_send_thr.get_id() && this->_send_thr.joinable()) this->_send_thr.join();
}

bool AUTDController::is_open() { return this->_link != nullptr && this->_link->is_open(); }

GeometryPtr AUTDController::geometry() noexcept { return this->_geometry; }

bool AUTDController::silent_mode() noexcept { return this->_silent_mode; }

size_t AUTDController::remaining_in_buffer() {
  return this->_send_gain_q.size() + this->_send_mod_q.size() + this->_build_gain_q.size() + this->_build_mod_q.size();
}

void AUTDController::SetSilentMode(bool silent) noexcept { this->_silent_mode = silent; }

void AUTDController::OpenWith(LinkPtr link) {
  this->Close();

  this->_link = link;
  this->_link->Open();
  if (this->_link->is_open())
    this->InitPipeline();
  else
    this->Close();
}

void AUTDController::CloseLink() { this->_link->Close(); }

void AUTDController::SendData(size_t size, std::unique_ptr<uint8_t[]> buf) { this->_link->Send(size, std::move(buf)); }
std::vector<uint8_t> AUTDController::ReadData(uint32_t buffer_len) { return this->_link->Read(buffer_len); }

size_t& AUTDController::mod_sent(ModulationPtr mod) { return mod->_sent; }
size_t& AUTDController::seq_sent(SequencePtr seq) { return seq->_sent; }
uint16_t AUTDController::seq_div(SequencePtr seq) { return seq->_sampl_freq_div; }

const uint16_t* AUTDController::gain_data_addr(GainPtr gain, int device_id) { return &gain->_data[device_id].at(0); }
}  // namespace autd
