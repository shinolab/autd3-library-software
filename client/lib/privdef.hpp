/*
 * File: privdef.hpp
 * Project: lib
 * Created Date: 07/06/2016
 * Author: Seki Inoue
 * -----
 * Last Modified: 19/10/2019
 * Modified By: Shun Suzuki (suzuki@hapis.k.u-tokyo.ac.jp)
 * -----
 * Copyright (c) 2016-2019 Hapis Lab. All rights reserved.
 * 
 */

#ifndef privdef_h
#define privdef_h

#include <stdint.h>
#include <array>

constexpr auto NUM_TRANS_IN_UNIT = 249;
constexpr auto NUM_TRANS_X = 18;
constexpr auto NUM_TRANS_Y = 14;
constexpr auto TRANS_SIZE_MM = 10.18f;
template <typename T>
constexpr auto IS_MISSING_TRANSDUCER(T X, T Y) { return (Y == 1 && (X == 1 || X == 2 || X == 16)); }

constexpr auto FPGA_CLOCK = 25600000;

constexpr auto ULTRASOUND_FREQUENCY_DEFAULT = 40000;
constexpr auto ULTRASOUND_WAVELENGTH = 8.5f;

constexpr auto MOD_SAMPLING_FREQ = 4000;
constexpr auto MOD_BUF_SIZE = 4000;
constexpr auto MOD_FRAME_SIZE = 124;

enum RxGlobalControlFlags
{
	LOOP_BEGIN = 1 << 0,
	LOOP_END = 1 << 1,
	SILENT = 1 << 3,
	MOD_RESET_MODE = 1 << 5,
};

struct RxGlobalHeader
{
	uint8_t msg_id;
	uint8_t control_flags;
	int8_t frequency_shift;
	uint8_t mod_size;
	uint8_t mod[MOD_FRAME_SIZE];
};

#endif /* privdef_h */
