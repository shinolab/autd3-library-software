// File: gain.hpp
// Project: include
// Created Date: 11/04/2018
// Author: Shun Suzuki
// -----
// Last Modified: 22/02/2020
// Modified By: Shun Suzuki (suzuki@hapis.k.u-tokyo.ac.jp)
// -----
// Copyright (c) 2018-2020 Hapis Lab. All rights reserved.
//

#pragma once

#define _USE_MATH_DEFINES
#include <math.h>

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
    auto d = asin(amp_i / 255.0) / M_PI;  //  duty (0 ~ 0.5)
    *amp_o = static_cast<uint8_t>(511 * d);
    *phase_o = static_cast<uint8_t>(static_cast<int>(phase_i + 64 - 128 * d) % 256);
  }

  std::mutex _mtx;
  bool _built;
  GeometryPtr _geometry;
  std::map<int, std::vector<uint16_t>> _data;

 public:
  static GainPtr Create();
  virtual void Build();
  void SetGeometry(const GeometryPtr &geometry) noexcept;
  GeometryPtr geometry() noexcept;
  std::map<int, std::vector<uint16_t>> data();
  bool built() noexcept;
};

using NullGain = Gain;

class PlaneWaveGain : public Gain {
 public:
  static GainPtr Create(Eigen::Vector3d direction);
  static GainPtr Create(Eigen::Vector3d direction, uint8_t amp);
  void Build() override;

 private:
  Eigen::Vector3d _direction;
  uint8_t _amp;
};

class FocalPointGain : public Gain {
 public:
  static GainPtr Create(Eigen::Vector3d point);
  static GainPtr Create(Eigen::Vector3d point, uint8_t amp);
  void Build() override;

 private:
  Eigen::Vector3d _point;
  uint8_t _amp = 0xff;
};

class BesselBeamGain : public Gain {
 public:
  static GainPtr Create(Eigen::Vector3d point, Eigen::Vector3d vec_n, double theta_z);
  static GainPtr Create(Eigen::Vector3d point, Eigen::Vector3d vec_n, double theta_z, uint8_t amp);
  void Build() override;

 private:
  Eigen::Vector3d _point;
  Eigen::Vector3d _vec_n;
  double _theta_z = 0;
  uint8_t _amp = 0xff;
};

class CustomGain : public Gain {
 public:
  static GainPtr Create(uint16_t *data, int data_length);
  void Build() override;

 private:
  std::vector<uint16_t> _rawdata;
};

class GroupedGain : public Gain {
 public:
  static GainPtr Create(std::map<int, autd::GainPtr> gainmap);
  void Build() override;

 private:
  std::map<int, autd::GainPtr> _gainmap;
};

class HoloGainSdp : public Gain {
 public:
  static GainPtr Create(Eigen::MatrixX3d foci, Eigen::VectorXd amp);
  void Build() override;

 protected:
  Eigen::MatrixX3d _foci;
  Eigen::VectorXd _amp;
};

using HoloGain = HoloGainSdp;

class MatlabGain : public Gain {
 public:
  static GainPtr Create(std::string filename, std::string varname);
  void Build() override;

 protected:
  std::string _filename, _varname;
};

class TransducerTestGain : public Gain {
 public:
  static GainPtr Create(int transducer_index, int amp, int phase);
  void Build() override;

 protected:
  int _xdcr_idx = 0;
  uint8_t _amp = 0;
  uint8_t _phase = 0;
};
}  // namespace autd
