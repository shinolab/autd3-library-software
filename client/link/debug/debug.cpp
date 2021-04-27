// File: debug_link.cpp
// Project: lib
// Created Date: 22/12/2020
// Author: Shun Suzuki
// -----
// Last Modified: 04/04/2021
// Modified By: Shun Suzuki (suzuki@hapis.k.u-tokyo.ac.jp)
// -----
// Copyright (c) 2020 Hapis Lab. All rights reserved.
//

#include "link/debug.hpp"

#include <bitset>
#include <cstring>
#include <memory>
#include <ostream>
#include <sstream>
#include <string>

#include "../lib/autd_logic.hpp"
#include "../lib/pre_def.hpp"
#include "consts.hpp"

namespace autd::link {

#define GEN_CFLAG_ARM(var) \
  case var:                \
    return #var;

#pragma warning(push)
#pragma warning(disable : 26812)
std::string ControlFlagBit2String(const RX_GLOBAL_CONTROL_FLAGS flag) {
  switch (flag) {
    GEN_CFLAG_ARM(LOOP_BEGIN)
    GEN_CFLAG_ARM(LOOP_END)
    GEN_CFLAG_ARM(MOD_BEGIN)
    GEN_CFLAG_ARM(SILENT)
    GEN_CFLAG_ARM(FORCE_FAN)
    GEN_CFLAG_ARM(SEQ_MODE)
    GEN_CFLAG_ARM(SEQ_BEGIN)
    GEN_CFLAG_ARM(SEQ_END)
  }
  return "Unknown flag";
}
#pragma warning(pop)

std::string Command2String(const uint8_t cmd) {
  using namespace internal;  // NOLINT
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

std::string ControlFlag2String(const uint8_t flags) {
  if (flags == 0) {
    return "None";
  }

  auto index = flags;
  uint8_t mask = 1;
  auto is_first = true;
  std::ostringstream oss;
  while (index) {
    if (index % 2 != 0) {
      if (!is_first) {
        oss << " | ";
      }
      oss << ControlFlagBit2String(static_cast<RX_GLOBAL_CONTROL_FLAGS>(flags & mask));
      is_first = false;
    }

    index = index >> 1;
    mask = static_cast<uint8_t>(mask << 1);
  }
  return oss.str();
}

LinkPtr DebugLink::Create(std::ostream &out) {
  LinkPtr link = std::make_unique<DebugLink>(out);
  return link;
}
DebugLink::DebugLink(std::ostream &out) : _out(out) {}

Result<bool, std::string> DebugLink::Open() {
  this->_out << "Call: Open()" << std::endl;
  _is_open = true;
  return Ok(true);
}

Result<bool, std::string> DebugLink::Close() {
  this->_out << "Call: Close()" << std::endl;
  _is_open = false;
  return Ok(true);
}

Result<bool, std::string> DebugLink::Send(const size_t size, const uint8_t *buf) {
  this->_out << "Call: Send()" << std::endl;

  _last_msg_id = buf[0];

  const auto *header = reinterpret_cast<const RxGlobalHeader *>(&buf[0]);
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

  const auto num_device = (size - sizeof(RxGlobalHeader)) / (2 * NUM_TRANS_IN_UNIT);
  auto idx = sizeof(RxGlobalHeader);
  if (num_device != 0) {
    for (size_t i = 0; i < num_device; i++) {
      this->_out << "Body[" << i << "]: " << std::hex;
      for (size_t j = 0; j < 2 * NUM_TRANS_IN_UNIT; j++) {
        this->_out << static_cast<int>(buf[idx + j]) << ", ";
      }
      idx += 2 * NUM_TRANS_IN_UNIT;
      this->_out << std::endl;
    }
  }

  return Ok(true);
}

Result<bool, std::string> DebugLink::Read(uint8_t *rx, const uint32_t buffer_len) {
  std::memset(rx, _last_msg_id, buffer_len);
  return Ok(true);
}

bool DebugLink::is_open() { return _is_open; }

}  // namespace autd::link
