// File: gain.hpp
// Project: include
// Created Date: 11/04/2018
// Author: Shun Suzuki
// -----
// Last Modified: 09/06/2020
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

#include "core.hpp"
#include "geometry.hpp"
#include "vector3.hpp"

namespace autd {
namespace gain {

inline void SignalDesign(uint8_t amp_i, uint8_t phase_i, uint8_t *const amp_o, uint8_t *const phase_o) noexcept {
  auto d = asin(amp_i / 255.0) / M_PI;  //  duty (0 ~ 0.5)
  *amp_o = static_cast<uint8_t>(511 * d);
  *phase_o = static_cast<uint8_t>(static_cast<int>(phase_i + 64 - 128 * d) % 256);
}

/**
 * @brief Gain controls the amplitude and phase of each transducer in the AUTD
 */
class Gain {
  friend class autd::AUTDController;
  friend class Geometry;

 public:
  /**
   * @brief Generate empty gain
   */
  static GainPtr Create();
  /**
   * @brief Calcurate amplitude and phase of each transducer
   */
  virtual void Build();
  /**
   * @brief Set AUTD Geometry which is required to build gain
   */
  void SetGeometry(const GeometryPtr &geometry) noexcept;
  /**
   * @brief Getter function for the data of amplitude and phase of each transducers
   * @details Each data is 16 bit unsigned integer, where MSB represents amplitude and LSB represents phase
   */
  std::map<int, std::vector<uint16_t>> data();

 protected:
  Gain() noexcept;
  bool _built;
  GeometryPtr _geometry;
  std::map<int, std::vector<uint16_t>> _data;
  GeometryPtr geometry() noexcept;
  bool built() noexcept;
};

/**
 * @brief Gain to create plane wave
 */
class PlaneWaveGain : public Gain {
 public:
  /**
   * @brief Generate function
   * @param[in] direction wave direction
   * @param[in] amp amplitude of the wave
   */
  static GainPtr Create(Vector3 direction, uint8_t amp = 0xff);
  void Build() override;

 private:
  Vector3 _direction = Vector3::unit_z();
  uint8_t _amp = 0xFF;
};

/**
 * @brief Gain to produce single focal point
 */
class FocalPointGain : public Gain {
 public:
  /**
   * @brief Generate function
   * @param[in] point focal point
   * @param[in] amp amplitude of the focus
   */
  static GainPtr Create(Vector3 point, uint8_t amp = 0xff);
  void Build() override;

 private:
  Vector3 _point;
  uint8_t _amp = 0xff;
};

/**
 * @brief Gain to produce Bessel Beam
 */
class BesselBeamGain : public Gain {
 public:
  /**
   * @brief Generate function
   * @param[in] point start point of the beam
   * @param[in] vec_n direction of the beam
   * @param[in] theta_z angle between the conical wavefront of the beam and the direction
   * @param[in] amp amplitude of the beam
   */
  static GainPtr Create(Vector3 point, Vector3 vec_n, double theta_z, uint8_t amp = 0xff);
  void Build() override;

 private:
  Vector3 _point = Vector3::zero();
  Vector3 _vec_n = Vector3::unit_z();
  double _theta_z = 0;
  uint8_t _amp = 0xff;
};

/**
 * @brief Gain that can set the phase and amplitude freely
 */
class CustomGain : public Gain {
 public:
  /**
   * @brief Generate function
   * @param[in] data pointer to data of amplitude and phase of each transducer
   * @param[in] data_length length of the data
   * @details The data length should be the same as the number of transducers you use. The data is 16 bit unsigned integer, where MSB represents
   * amplitude and LSB represents phase
   */
  static GainPtr Create(uint16_t *data, int data_length);
  void Build() override;

 private:
  std::vector<uint16_t> _rawdata;
};

/**
 * @brief Gain to group some gains
 */
class GroupedGain : public Gain {
 public:
  /**
   * @brief Generate function
   * @param[in] gainmap ｍap from group ID to gain
   * @details group ID must be specified in Geometry::AddDevice() in advance
   */
  static GainPtr Create(std::map<int, GainPtr> gainmap);
  void Build() override;

 private:
  std::map<int, GainPtr> _gainmap;
};

/**
 * @brief Gain to produce multiple focal points
 */
class HoloGainSdp : public Gain {
 public:
  /**
   * @brief Generate function
   * @param[in] point focal points
   * @param[in] amps amplitudes of the foci
   */
  static GainPtr Create(std::vector<Vector3> foci, std::vector<double> amps);
  void Build() override;

 protected:
  std::vector<Vector3> _foci;
  std::vector<double> _amps;
};

/**
 * @brief Gain created from Matlab mat file
 */
class MatlabGain : public Gain {
 public:
  /**
   * @brief Generate function
   * @param[in] filename mat file path
   * @param[in] varname variable name in mat file
   */
  static GainPtr Create(std::string filename, std::string varname);
  void Build() override;

 protected:
  std::string _filename, _varname;
};

/**
 * @brief Gain to test a transducer
 */
class TransducerTestGain : public Gain {
 public:
  /**
   * @brief Generate function
   * @param[in] transducer_index index of the transducer
   * @param[in] amp amplitude of the transducer
   * @param[in] phase phase of the phase
   */
  static GainPtr Create(int transducer_index, int amp, int phase);
  void Build() override;

 protected:
  int _xdcr_idx = 0;
  uint8_t _amp = 0;
  uint8_t _phase = 0;
};
}  // namespace gain
}  // namespace autd
