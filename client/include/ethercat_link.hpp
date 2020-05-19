// File: ethercat_link.hpp
// Project: lib
// Created Date: 01/06/2016
// Author: Seki Inoue
// -----
// Last Modified: 19/05/2020
// Modified By: Shun Suzuki (suzuki@hapis.k.u-tokyo.ac.jp)
// -----
// Copyright (c) 2016-2020 Hapis Lab. All rights reserved.
//

#pragma once

#include <stdio.h>

#include <memory>
#include <string>
#include <vector>

#if WIN32
#include <codeanalysis\warnings.h>
#pragma warning(push)
#pragma warning(disable : ALL_CODE_ANALYSIS_WARNINGS)
#endif
#include <AdsLib.h>
#if WIN32
#pragma warning(pop)
#endif

#include "core.hpp"
#include "link.hpp"

#ifdef _WINDOWS
#define NOMINMAX
#include <Windows.h>
#include <winnt.h>
#else
typedef void *HMODULE;
#endif

namespace autd {
class EthercatLink : public Link {
 public:
  static LinkPtr Create(std::string ipv4addr);
  static LinkPtr Create(std::string ipv4addr, std::string ams_net_id);

  ~EthercatLink() override {};

 protected:
  void Open() override;
  void Close() override;
  void Send(size_t size, std::unique_ptr<uint8_t[]> buf) override;
  std::vector<uint8_t> Read(uint32_t buffer_len) override;
  bool is_open() final;
  bool CalibrateModulation() final;

 protected:
  std::string _ams_net_id;
  std::string _ipv4addr;
  long _port = 0L;  // NOLINT
  AmsNetId _netId;
};

class LocalEthercatLink : public EthercatLink {
 public:
  static LinkPtr Create();
  ~LocalEthercatLink() override {}

 protected:
  void Open() final;
  void Close() final;
  void Send(size_t size, std::unique_ptr<uint8_t[]> buf) final;
  std::vector<uint8_t> Read(uint32_t buffer_len) final;

 private:
  HMODULE lib = nullptr;
};
}  // namespace autd
