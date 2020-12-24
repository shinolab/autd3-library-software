// File: debug_link.cpp
// Project: lib
// Created Date: 22/12/2020
// Author: Shun Suzuki
// -----
// Last Modified: 22/12/2020
// Modified By: Shun Suzuki (suzuki@hapis.k.u-tokyo.ac.jp)
// -----
// Copyright (c) 2020 Hapis Lab. All rights reserved.
//

#include "debug_link.hpp"

#include <algorithm>
#include <bitset>
#include <chrono>
#include <cstring>
#include <iostream>
#include <memory>
#include <ostream>
#include <sstream>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "autd_logic.hpp"
#include "consts.hpp"
#include "privdef.hpp"

namespace autd::link {

#define GEN_CFLAG_ARM(var) \
  case var:                \
    return #var;

#pragma warning(push)
#pragma warning(disable : 26812)
std::string ControlFlagBit2String(RxGlobalControlFlags flag) {
  switch (flag) {
    GEN_CFLAG_ARM(LOOP_BEGIN)
    GEN_CFLAG_ARM(LOOP_END)
    GEN_CFLAG_ARM(MOD_BEGIN)
    GEN_CFLAG_ARM(SILENT)
    GEN_CFLAG_ARM(FORCE_FAN)
    GEN_CFLAG_ARM(SEQ_MODE)
    GEN_CFLAG_ARM(SEQ_BEGIN)
    GEN_CFLAG_ARM(SEQ_END)
    default:
      return "Unknown flag";
  }
}
#pragma warning(pop)

std::string Command2String(uint8_t cmd) {
  using namespace autd::_internal;  // NOLINT
  switch (cmd) {
    GEN_CFLAG_ARM(CMD_OP)
    GEN_CFLAG_ARM(CMD_BRAM_WRITE)
    GEN_CFLAG_ARM(CMD_READ_CPU_VER_LSB)
    GEN_CFLAG_ARM(CMD_READ_CPU_VER_MSB)
    GEN_CFLAG_ARM(CMD_READ_FPGA_VER_LSB)
    GEN_CFLAG_ARM(CMD_READ_FPGA_VER_MSB)
    GEN_CFLAG_ARM(CMD_SEQ_MODE)
    GEN_CFLAG_ARM(CMD_INIT_REF_CLOCK)
    GEN_CFLAG_ARM(CMD_CALIB_SEQ_CLOCK)
    GEN_CFLAG_ARM(CMD_CLEAR)
    default:
      return "Unknown command";
  }
}

std::string ControlFlag2String(uint8_t flags) {
  if (flags == 0) {
    return "None";
  }

  uint8_t index = flags;
  uint8_t mask = 1;
  bool isFirst = true;
  std::ostringstream oss;
  while (index) {
    if (index % 2 != 0) {
      if (!isFirst) {
        oss << " | ";
      }
      oss << ControlFlagBit2String(static_cast<RxGlobalControlFlags>(flags & mask));
      isFirst = false;
    }

    index = index >> 1;
    mask = mask << 1;
  }
  return oss.str();
}

LinkPtr DebugLink::Create(std::ostream &out) {
  auto link = std::make_shared<DebugLink>(out);
  return link;
}
DebugLink::DebugLink(std::ostream &out) : _out(out) {}

void DebugLink::Open() {
  this->_out << "Call: Open()" << std::endl;
  _is_open = true;
}

void DebugLink::Close() {
  this->_out << "Call: Close()" << std::endl;
  _is_open = false;
}

void DebugLink::Send(size_t size, std::unique_ptr<uint8_t[]> buf) {
  this->_out << "Call: Send()" << std::endl;

  _last_msg_id = buf[0];

  RxGlobalHeader *header = reinterpret_cast<RxGlobalHeader *>(&buf[0]);
  this->_out << "Header:" << std::endl;
  this->_out << "\tmsg_id   : " << std::hex << static_cast<int>(header->msg_id) << std::endl;
  this->_out << "\tflag     : " << ControlFlag2String(header->control_flags) << std::endl;
  this->_out << "\tcommand  : " << Command2String(header->command) << std::endl;
  if (header->mod_size != 0) {
    this->_out << "\tmod_size : " << std::dec << static_cast<int>(header->mod_size) << std::endl;
    this->_out << "\tmod      : " << std::hex;
    for (uint8_t i = 0; i < header->mod_size; i++) {
      this->_out << static_cast<int>(header->mod[i]) << ", ";
    }
    this->_out << std::endl;
  }
  if (header->seq_size != 0) {
    this->_out << "\tseq_size  : " << std::dec << static_cast<int>(header->seq_size) << std::endl;
    this->_out << "\tseq_div  : " << static_cast<int>(header->seq_div) << std::endl;
  }

  auto num_device = (size - sizeof(RxGlobalHeader)) / (2 * NUM_TRANS_IN_UNIT);
  auto idx = sizeof(RxGlobalHeader);
  if (num_device != 0) {
    for (auto i = 0; i < num_device; i++) {
      this->_out << "Body[" << i << "]: " << std::hex;
      for (auto j = 0; j < 2 * NUM_TRANS_IN_UNIT; j++) {
        this->_out << static_cast<int>(buf[idx + j]) << ", ";
      }
      idx += 2 * NUM_TRANS_IN_UNIT;
      this->_out << std::endl;
    }
  }
}

std::vector<uint8_t> DebugLink::Read(uint32_t buffer_len) { return std::vector<uint8_t>(buffer_len, _last_msg_id); }

bool DebugLink::is_open() { return _is_open; }

};  // namespace autd::link
