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
  this->_out << "\tmsg_id:" << static_cast<int>(header->msg_id) << std::endl;
  this->_out << "\tflag  :" << ControlFlag2String(header->control_flags) << std::endl;
}

std::vector<uint8_t> DebugLink::Read(uint32_t buffer_len) { return std::vector<uint8_t>(buffer_len, _last_msg_id); }

bool DebugLink::is_open() { return _is_open; }

};  // namespace autd::link
