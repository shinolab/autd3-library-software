// File: sequence.hpp
// Project: core
// Created Date: 14/05/2021
// Author: Shun Suzuki
// -----
// Last Modified: 14/12/2021
// Modified By: Shun Suzuki (suzuki@hapis.k.u-tokyo.ac.jp)
// -----
// Copyright (c) 2021 Hapis Lab. All rights reserved.
//

#pragma once

#include <algorithm>
#include <string>
#include <vector>

#include "exception.hpp"
#include "gain.hpp"
#include "geometry.hpp"
#include "type_hints.hpp"

namespace autd::core {
class PointSequence;
class GainSequence;

class Sequence : public IDatagramBody {
 public:
  Sequence() : _freq_div_ratio(1), _wait_on_sync(false) {}
  ~Sequence() override = default;
  Sequence(const Sequence& v) noexcept = delete;
  Sequence& operator=(const Sequence& obj) = delete;
  Sequence(Sequence&& obj) = default;
  Sequence& operator=(Sequence&& obj) = default;

  [[nodiscard]] virtual size_t size() const = 0;

  /**
   * @brief Set frequency of the sequence
   * @param[in] freq Frequency of the sequence
   * @details Sequence mode has some constraints, which determine the actual frequency of the sequence.
   * @return double Actual frequency of sequence
   */
  double set_frequency(const double freq) {
    const auto sample_freq = static_cast<double>(this->size()) * freq;
    this->_freq_div_ratio =
        std::clamp<size_t>(static_cast<size_t>(std::round(static_cast<double>(SEQ_BASE_FREQ) / sample_freq)), 1, SEQ_SAMPLING_FREQ_DIV_MAX);
    return this->frequency();
  }

  /**
   * @return frequency of sequence
   */
  [[nodiscard]] double frequency() const { return this->sampling_freq() / static_cast<double>(this->size()); }

  /**
   * @return period of sequence in micro seconds
   */
  [[nodiscard]] size_t period_us() const { return this->sampling_period_us() * this->size(); }

  /**
   * The sampling period is limited to an integer multiple of 25us. Therefore, the sampling frequency must be 40kHz/N, where N=1, 2, ...,
   * autd::core::SEQ_SAMPLING_FREQ_DIV_MAX.
   * @return double Sampling frequency of sequence
   */
  [[nodiscard]] double sampling_freq() const { return static_cast<double>(SEQ_BASE_FREQ) / static_cast<double>(this->_freq_div_ratio); }

  /**
   * @return sampling period of sequence in micro seconds
   */
  [[nodiscard]] size_t sampling_period_us() const noexcept { return _freq_div_ratio * 1000000 / SEQ_BASE_FREQ; }

  /**
   * The sampling frequency division ratio means the autd::core::SEQ_BASE_FREQ/(sampling frequency) = (sampling period)/25us.
   * @return size_t Sampling frequency division ratio
   * \details  The value must be in 1, 2, ..., autd::core::SEQ_SAMPLING_FREQ_DIV_MAX.
   */
  size_t& sampling_freq_div_ratio() noexcept { return this->_freq_div_ratio; }

  /**
   * The sampling frequency division ratio means the autd::core::SEQ_BASE_FREQ/(sampling frequency) = (sampling period)/25us.
   * @return size_t Sampling frequency division ratio
   * \details  The value must be in 1, 2, ..., autd::core::SEQ_SAMPLING_FREQ_DIV_MAX.
   */
  [[nodiscard]] size_t sampling_freq_div_ratio() const noexcept { return this->_freq_div_ratio; }

  /**
   * @brief If true, the output will be start after synchronization.
   */
  [[nodiscard]] bool wait_on_sync() const noexcept { return this->_wait_on_sync; }

  /**
   * @brief If true, the output will be start after synchronization.
   */
  bool& wait_on_sync() noexcept { return this->_wait_on_sync; }

 protected:
  size_t _freq_div_ratio;
  bool _wait_on_sync;
};

/**
 * @brief PointSequence provides a function to display the focus sequentially and periodically.
 * @details PointSequence uses a timer on the FPGA to ensure that the focus is precisely timed.
 * PointSequence currently has the following three limitations.
 * 1. The maximum number of control points is autd::core::POINT_SEQ_BUFFER_SIZE_MAX.
 * 2. The sampling interval of control points is an integer multiple of 25us and less than or equal to 25us x autd::core::SEQ_SAMPLING_FREQ_DIV_MAX.
 * 3. Only a single focus can be displayed at a certain moment.
 */
class PointSequence : virtual public Sequence {
  struct SeqFocus {
    SeqFocus() = default;

    void set(const int32_t x, const int32_t y, const int32_t z, const uint8_t duty) {
      _buf[0] = x & 0xFFFF;             // x 0-15 bit
      uint16_t tmp = x >> 16 & 0x0001;  // x 16th bit
      tmp |= x >> 30 & 0x0002;          // x sign bit
      tmp |= y << 2 & 0xFFFC;           // y 0-13 bit
      _buf[1] = tmp;
      tmp = y >> 14 & 0x0007;   // y 14-16 bit
      tmp |= y >> 28 & 0x0008;  // y sign bit
      tmp |= z << 4 & 0xFFF0;   // z 0-11 bit
      _buf[2] = tmp;
      tmp = z >> 12 & 0x001F;     // z 12-16 bit
      tmp |= z >> 26 & 0x0020;    // z sign bit
      tmp |= duty << 6 & 0x3FC0;  // duty
      _buf[3] = tmp;
    }

   private:
    uint16_t _buf[4];
  };

 public:
  PointSequence() noexcept : Sequence(), _sent(0) {}

  [[nodiscard]] size_t size() const override { return this->_control_points.size(); }

  /**
   * @brief Add control point
   * @param[in] point control point
   * @param[in] duty duty ratio
   */
  void add(const Vector3& point, const uint8_t duty = 0xFF) {
    if (this->_control_points.size() + 1 > POINT_SEQ_BUFFER_SIZE_MAX)
      throw exception::SequenceBuildError(
          std::string("Point sequence buffer overflow. Maximum available buffer size is " + std::to_string(POINT_SEQ_BUFFER_SIZE_MAX)));

    this->_control_points.emplace_back(point);
    this->_duties.emplace_back(duty);
  }

  /**
   * @brief Add control points
   * @param[in] points control points
   * @param[in] duties duty ratios
   * @details duties.resize(points.size(), 0xFF) will be called.
   */
  void add(const std::vector<Vector3>& points, std::vector<uint8_t>& duties) {
    if (this->_control_points.size() + points.size() > POINT_SEQ_BUFFER_SIZE_MAX)
      throw exception::SequenceBuildError(
          std::string("Point sequence buffer overflow. Maximum available buffer size is " + std::to_string(POINT_SEQ_BUFFER_SIZE_MAX)));

    this->_control_points.reserve(this->_control_points.size() + points.size());
    this->_control_points.insert(this->_control_points.end(), points.begin(), points.end());

    duties.resize(points.size(), 0xFF);
    this->_duties.reserve(this->_duties.size() + duties.size());
    this->_duties.insert(this->_duties.end(), duties.begin(), duties.end());
  }

  /**
   * @return std::vector<Vector3> Control points of the sequence
   */
  [[nodiscard]] const std::vector<Vector3>& control_points() const { return this->_control_points; }

  /**
   * @return std::vector<uint8_t> Duty ratios of the sequence
   */
  [[nodiscard]] const std::vector<uint8_t>& duties() const { return this->_duties; }

  void init() override { _sent = 0; }

  void pack(const Geometry& geometry, TxDatagram& tx) override {
    auto* header = tx.header();

    if (_wait_on_sync) header->cpu_ctrl_flags |= WAIT_ON_SYNC;
    header->fpga_ctrl_flags |= OUTPUT_ENABLE;
    header->fpga_ctrl_flags |= SEQ_MODE;
    header->fpga_ctrl_flags &= ~SEQ_GAIN_MODE;

    if (is_finished()) return;

    tx.num_bodies() = geometry.num_devices();

    size_t offset = 1;
    header->cpu_ctrl_flags |= WRITE_BODY;
    if (_sent == 0) {
      header->cpu_ctrl_flags |= SEQ_BEGIN;
      for (const auto& device : geometry) {
        auto* cursor = reinterpret_cast<uint16_t*>(tx.body(device.id()));
        cursor[1] = static_cast<uint16_t>(_freq_div_ratio - 1);
        cursor[2] = static_cast<uint16_t>(geometry.wavelength() * 1000);
      }
      offset += 4;
    }
    const auto send_size = std::clamp(_control_points.size() - _sent, size_t{0}, sizeof(uint16_t) * (NUM_TRANS_IN_UNIT - offset) / sizeof(SeqFocus));
    if (_sent + send_size >= _control_points.size()) header->cpu_ctrl_flags |= SEQ_END;

    const auto fixed_num_unit = 256.0 / geometry.wavelength();
    for (const auto& device : geometry) {
      auto* cursor = reinterpret_cast<uint16_t*>(tx.body(device.id()));
      cursor[0] = static_cast<uint16_t>(send_size);
      auto* focus_cursor = reinterpret_cast<SeqFocus*>(&cursor[offset]);
      for (size_t i = _sent; i < _sent + send_size; i++, focus_cursor++) {
        const auto v = (device.to_local_position(_control_points[i]) * fixed_num_unit).cast<int32_t>();
        focus_cursor->set(v.x(), v.y(), v.z(), _duties[i]);
      }
    }
    _sent += send_size;
  }

  [[nodiscard]] bool is_finished() const override { return _sent == _control_points.size(); }

  class StreamCommaInputPS {
   public:
    explicit StreamCommaInputPS(PointSequence& seq) : _seq(seq) {}
    ~StreamCommaInputPS() = default;
    StreamCommaInputPS(const StreamCommaInputPS& v) noexcept = delete;
    StreamCommaInputPS& operator=(const StreamCommaInputPS& obj) = delete;
    StreamCommaInputPS(StreamCommaInputPS&& obj) = default;
    StreamCommaInputPS& operator=(StreamCommaInputPS&& obj) = delete;

    StreamCommaInputPS& operator,(const Vector3& point) {
      _seq.add(point);
      return *this;
    }

    StreamCommaInputPS& operator<<(const Vector3& point) {
      _seq.add(point);
      return *this;
    }

   private:
    PointSequence& _seq;
  };

  StreamCommaInputPS operator<<(const Vector3& point) {
    this->add(point);
    return StreamCommaInputPS{*this};
  }

 private:
  std::vector<Vector3> _control_points;
  std::vector<uint8_t> _duties;
  size_t _sent;
};

enum class GAIN_MODE : uint16_t {
  DUTY_PHASE_FULL = 0x0001,
  PHASE_FULL = 0x0002,
  PHASE_HALF = 0x0004,
};

/**
 * @brief GainSequence provides a function to display Gain sequentially and periodically.
 * @details GainSequence uses a timer on the FPGA to ensure that Gain is precisely timed.
 * GainSequence currently has the following three limitations.
 * 1. The maximum number of gains is autd::core::GAIN_SEQ_BUFFER_SIZE_MAX.
 * 2. The sampling interval of gains is an integer multiple of 25us and less than 25us x autd::core::SEQ_SAMPLING_FREQ_DIV_MAX.
 */
class GainSequence final : virtual public Sequence {
  struct PhaseDrive {
    PhaseDrive() = default;
    void set(const uint8_t phase0, const uint8_t phase1) {
      _phase0 = phase0;
      _phase1 = phase1;
    }

   private:
    uint8_t _phase0;
    uint8_t _phase1;
  };
  struct HalfPhaseDrive {
    HalfPhaseDrive() = default;
    void set(const uint8_t phase0, const uint8_t phase1, const uint8_t phase2, const uint8_t phase3) {
      _phase01 = (phase1 & 0xF0) | ((phase0 >> 4) & 0x0F);
      _phase23 = (phase3 & 0xF0) | ((phase2 >> 4) & 0x0F);
    }

   private:
    uint8_t _phase01;
    uint8_t _phase23;
  };

 public:
  explicit GainSequence(const Geometry& geometry) noexcept : GainSequence(geometry, GAIN_MODE::DUTY_PHASE_FULL) {}
  explicit GainSequence(const Geometry& geometry, const GAIN_MODE gain_mode) noexcept
      : Sequence(), _geometry(geometry), _gain_mode(gain_mode), _sent(0) {}

  [[nodiscard]] size_t size() const override { return this->_gain_drives.size(); }

  /**
   * @brief Add gain
   * @param[in] gain gain
   */
  template <class T>
  std::enable_if_t<is_gain_v<T>> add(T&& gain) {
    if (this->_gain_drives.size() + 1 > GAIN_SEQ_BUFFER_SIZE_MAX)
      throw exception::SequenceBuildError(
          std::string("Gain sequence buffer overflow. Maximum available buffer size is " + std::to_string(GAIN_SEQ_BUFFER_SIZE_MAX)));

    Gain& g = to_gain(gain);

    g.calc(_geometry);

    this->_gain_drives.emplace_back(g.data());
  }

  /**
   * @return GAIN_MODE
   */
  GAIN_MODE& gain_mode() { return this->_gain_mode; }
  /**
   * @return GAIN_MODE
   */
  [[nodiscard]] GAIN_MODE gain_mode() const { return this->_gain_mode; }

  void init() override { _sent = 0; }

  void pack(const Geometry& geometry, TxDatagram& tx) override {
    auto* header = tx.header();

    if (_wait_on_sync) header->cpu_ctrl_flags |= WAIT_ON_SYNC;
    header->fpga_ctrl_flags |= OUTPUT_ENABLE;
    header->fpga_ctrl_flags |= SEQ_MODE;
    header->fpga_ctrl_flags |= SEQ_GAIN_MODE;

    if (is_finished()) return;

    tx.num_bodies() = geometry.num_devices();

    header->cpu_ctrl_flags |= WRITE_BODY;
    const auto sent = static_cast<size_t>(_gain_mode);
    if (_sent == 0) {
      header->cpu_ctrl_flags |= SEQ_BEGIN;
      for (const auto& device : geometry) {
        auto* cursor = reinterpret_cast<uint16_t*>(tx.body(device.id()));
        cursor[0] = static_cast<uint16_t>(sent);
        cursor[1] = static_cast<uint16_t>(_freq_div_ratio - 1);
        cursor[2] = static_cast<uint16_t>(_gain_drives.size());
      }
      _sent += 1;
      return;
    }

    if (_sent + sent > _gain_drives.size()) header->cpu_ctrl_flags |= SEQ_END;

    const auto gain_idx = _sent - 1;
    switch (_gain_mode) {
      case GAIN_MODE::DUTY_PHASE_FULL: {
        auto* cursor = reinterpret_cast<Drive*>(tx.body(0));
        std::memcpy(cursor, _gain_drives[gain_idx].data(), _gain_drives[gain_idx].size() * sizeof(Drive));
      } break;
      case GAIN_MODE::PHASE_FULL:
        for (const auto& dev : geometry) {
          auto* cursor = reinterpret_cast<PhaseDrive*>(tx.body(dev.id()));
          for (const auto& trans : dev)
            cursor[trans.id()].set(_gain_drives[gain_idx][trans.id()].phase,
                                   gain_idx + 1 >= _gain_drives.size() ? 0x00 : _gain_drives[gain_idx + 1][trans.id()].phase);
        }
        break;
      case GAIN_MODE::PHASE_HALF:
        for (const auto& dev : geometry) {
          auto* cursor = reinterpret_cast<HalfPhaseDrive*>(tx.body(dev.id()));
          for (const auto& trans : dev)
            cursor[trans.id()].set(_gain_drives[gain_idx][trans.id()].phase,
                                   gain_idx + 1 >= _gain_drives.size() ? 0x00 : _gain_drives[gain_idx + 1][trans.id()].phase,
                                   gain_idx + 2 >= _gain_drives.size() ? 0x00 : _gain_drives[gain_idx + 2][trans.id()].phase,
                                   gain_idx + 3 >= _gain_drives.size() ? 0x00 : _gain_drives[gain_idx + 3][trans.id()].phase);
        }
        break;
    }
    _sent += sent;
  }

  [[nodiscard]] bool is_finished() const override { return _sent == _gain_drives.size() + 1; }

  class StreamCommaInputGS {
   public:
    explicit StreamCommaInputGS(GainSequence& cnt) : _cnt(cnt) {}
    ~StreamCommaInputGS() = default;
    StreamCommaInputGS(const StreamCommaInputGS& v) noexcept = delete;
    StreamCommaInputGS& operator=(const StreamCommaInputGS& obj) = delete;
    StreamCommaInputGS(StreamCommaInputGS&& obj) = default;
    StreamCommaInputGS& operator=(StreamCommaInputGS&& obj) = delete;

    template <class T>
    std::enable_if_t<is_gain_v<T>, StreamCommaInputGS&> operator,(T&& gain) {
      _cnt.add(gain);
      return *this;
    }

    template <class T>
    std::enable_if_t<is_gain_v<T>, StreamCommaInputGS&> operator<<(T&& gain) {
      _cnt.add(gain);
      return *this;
    }

   private:
    GainSequence& _cnt;
  };

  template <class T>
  std::enable_if_t<is_gain_v<T>, StreamCommaInputGS> operator<<(T&& gain) {
    this->add(gain);
    return StreamCommaInputGS{*this};
  }

 private:
  const Geometry& _geometry;
  std::vector<std::vector<Drive>> _gain_drives;
  GAIN_MODE _gain_mode;
  size_t _sent;
};

}  // namespace autd::core
