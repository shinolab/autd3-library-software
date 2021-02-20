// File: gain.hpp
// Project: include
// Created Date: 11/04/2018
// Author: Shun Suzuki
// -----
// Last Modified: 20/02/2021
// Modified By: Shun Suzuki (suzuki@hapis.k.u-tokyo.ac.jp)
// -----
// Copyright (c) 2018-2020 Hapis Lab. All rights reserved.
//

#pragma once

#include <map>

#include "autd_types.hpp"
#include "consts.hpp"
#include "geometry.hpp"

namespace autd {
namespace gain {
class Gain;
}
using GainPtr = std::shared_ptr<gain::Gain>;
}  // namespace autd

namespace autd::gain {

inline Float PosMod(const Float a, const Float b) { return a - floor(a / b) * b; }

inline uint8_t AdjustAmp(const Float amp) noexcept {
  const auto d = asin(amp) / PI;  //  duty (0 ~ 0.5)
  return static_cast<uint8_t>(511 * d);
}

inline void CheckAndInit(const GeometryPtr& geometry, std::vector<AUTDDataArray>* data) {
  assert(geometry != nullptr);

  data->clear();

  const auto num_device = geometry->num_devices();
  data->resize(num_device);
}

/**
 * @brief Gain controls the amplitude and phase of each transducer in the AUTD
 */
class Gain {
 public:
  /**
   * @brief Generate empty gain
   */
  static GainPtr Create();
  /**
   * @brief Calculate amplitude and phase of each transducer
   */
  virtual void Build();
  /**
   * @brief Set AUTD Geometry which is required to build gain
   */
  void SetGeometry(const GeometryPtr& geometry) noexcept;
  /**
   * @brief Get AUTD Geometry
   */
  [[nodiscard]] GeometryPtr geometry() const noexcept;
  /**
   * @brief Getter function for the data of amplitude and phase of each transducers
   * @details Each data is 16 bit unsigned integer, where MSB represents amplitude and LSB represents phase
   */
  std::vector<AUTDDataArray>& data();

  Gain() noexcept;
  virtual ~Gain() = default;
  Gain(const Gain& v) noexcept = default;
  Gain& operator=(const Gain& obj) = default;
  Gain(Gain&& obj) = default;
  Gain& operator=(Gain&& obj) = default;

 protected:
  explicit Gain(std::vector<AUTDDataArray> data) noexcept;
  bool _built;
  GeometryPtr _geometry;
  std::vector<AUTDDataArray> _data;
  [[nodiscard]] bool built() const noexcept;
};

using NullGain = Gain;

/**
 * @brief Gain to group some gains
 */
class GroupedGain final : public Gain {
 public:
  /**
   * @brief Generate function
   * @param[in] gain_map ｍap from group ID to gain
   * @details group ID must be specified in Geometry::AddDevice() in advance
   */
  static GainPtr Create(const std::map<size_t, GainPtr>& gain_map);
  void Build() override;
  explicit GroupedGain(std::map<size_t, GainPtr> gain_map) : Gain(), _gain_map(std::move(gain_map)) {}
  ~GroupedGain() override = default;
  GroupedGain(const GroupedGain& v) noexcept = default;
  GroupedGain& operator=(const GroupedGain& obj) = default;
  GroupedGain(GroupedGain&& obj) = default;
  GroupedGain& operator=(GroupedGain&& obj) = default;

 private:
  std::map<size_t, GainPtr> _gain_map;
};
}  // namespace autd::gain
