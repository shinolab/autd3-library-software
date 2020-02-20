// File: gain.hpp
// Project: include
// Created Date: 11/04/2018
// Author: Shun Suzuki
// -----
// Last Modified: 18/02/2020
// Modified By: Shun Suzuki (suzuki@hapis.k.u-tokyo.ac.jp)
// -----
// Copyright (c) 2018-2020 Hapis Lab. All rights reserved.
//

#pragma once

#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#if WIN32
#include <codeanalysis\warnings.h>
#pragma warning(push)
#pragma warning(disable : ALL_CODE_ANALYSIS_WARNINGS)
#endif
#include <Eigen/Core>
#if WIN32
#pragma warning(pop)
#endif

#include "core.hpp"
#include "geometry.hpp"

constexpr auto M_PIf = 3.14159265f;

namespace autd {
class Gain;

#if DLL_FOR_CAPI
using GainPtr = Gain *;
#else
using GainPtr = std::shared_ptr<Gain>;
#endif

class Gain {
  friend class AUTDController;
  friend class Geometry;

 protected:
  Gain() noexcept;
  inline void SignalDesign(uint8_t amp_i, uint8_t phase_i, uint8_t *const amp_o, uint8_t *const phase_o) noexcept {
    auto d = asin(amp_i / 255.0) / M_PIf;  //  duty (0 ~ 0.5)
    *amp_o = static_cast<uint8_t>(511 * d);
    *phase_o = static_cast<uint8_t>(static_cast<int>(phase_i + 64 - 128 * d) % 256);
  }

  std::mutex _mtx;
  bool _built;
  GeometryPtr _geometry;
  std::map<int, std::vector<uint16_t>> _data;

 public:
  static GainPtr Create();
  virtual void build();
  void SetGeometry(const GeometryPtr &geometry) noexcept;
  GeometryPtr geometry() noexcept;
  std::map<int, std::vector<uint16_t>> data();
  bool built() noexcept;
};

using NullGain = Gain;

class PlaneWaveGain : public Gain {
 public:
  static GainPtr Create(Eigen::Vector3f direction);
  static GainPtr Create(Eigen::Vector3f direction, uint8_t amp);
  void build() override;

 private:
  Eigen::Vector3f _direction;
  uint8_t _amp;
};

class FocalPointGain : public Gain {
 public:
  static GainPtr Create(Eigen::Vector3f point);
  static GainPtr Create(Eigen::Vector3f point, uint8_t amp);
  void build() override;

 private:
  Eigen::Vector3f _point;
  uint8_t _amp = 0xff;
};

class BesselBeamGain : public Gain {
 public:
  static GainPtr Create(Eigen::Vector3f point, Eigen::Vector3f vec_n, float theta_z);
  static GainPtr Create(Eigen::Vector3f point, Eigen::Vector3f vec_n, float theta_z, uint8_t amp);
  void build() override;

 private:
  Eigen::Vector3f _point;
  Eigen::Vector3f _vec_n;
  float _theta_z = 0;
  uint8_t _amp = 0xff;
};

class CustomGain : public Gain {
 public:
  static GainPtr Create(uint16_t *data, int dataLength);
  void build() override;

 private:
  std::vector<uint16_t> _rawdata;
};

class GroupedGain : public Gain {
 public:
  static GainPtr Create(std::map<int, autd::GainPtr> gainmap);
  void build() override;

 private:
  std::map<int, autd::GainPtr> _gainmap;
};

class HoloGainSdp : public Gain {
 public:
  static GainPtr Create(Eigen::MatrixX3f foci, Eigen::VectorXf amp);
  void build() override;

 protected:
  Eigen::MatrixX3f _foci;
  Eigen::VectorXf _amp;
};

using HoloGain = HoloGainSdp;

class MatlabGain : public Gain {
 public:
  static GainPtr Create(std::string filename, std::string varname);
  void build() override;

 protected:
  std::string _filename, _varname;
};

class TransducerTestGain : public Gain {
 public:
  static GainPtr Create(int transducer_index, int amp, int phase);
  void build() override;

 protected:
  int _xdcr_idx = 0;
  uint8_t _amp = 0;
  uint8_t _phase = 0;
};
}  // namespace autd
