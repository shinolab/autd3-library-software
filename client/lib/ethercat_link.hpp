/*
 * File: ethercat_link.hpp
 * Project: lib
 * Created Date: 01/06/2016
 * Author: Seki Inoue
 * -----
 * Last Modified: 18/02/2020
 * Modified By: Shun Suzuki (suzuki@hapis.k.u-tokyo.ac.jp)
 * -----
 * Copyright (c) 2016-2019 Hapis Lab. All rights reserved.
 *
 */

#pragma once

#include <codeanalysis\warnings.h>
#include <stdio.h>

#include <memory>
#include <string>

#pragma warning(push)
#pragma warning(disable : ALL_CODE_ANALYSIS_WARNINGS)
#include <AdsLib.h>
#pragma warning(pop)
#include "link.hpp"

#ifdef _WINDOWS
#define NOMINMAX
#include <Windows.h>
#include <winnt.h>
#else
typedef void *HMODULE;
#endif

namespace autd {
namespace internal {
class EthercatLink : public Link {
 public:
  virtual void Open(std::string location);
  virtual void Open(std::string ams_net_id, std::string ipv4addr);
  virtual void Close();
  virtual void Send(size_t size, std::unique_ptr<uint8_t[]> buf);
  bool isOpen();
  bool CalibrateModulation();

 protected:
  long _port = 0L;  // NOLINT
  AmsNetId _netId;
};

class LocalEthercatLink : public EthercatLink {
 public:
  void Open(std::string location = "");
  void Close();
  void Send(size_t size, std::unique_ptr<uint8_t[]> buf);

 private:
  HMODULE lib = NULL;
};
}  // namespace internal
}  // namespace autd
