/*
 * File: controller.hpp
 * Project: include
 * Created Date: 11/04/2018
 * Author: Shun Suzuki
 * -----
 * Last Modified: 10/02/2020
 * Modified By: Shun Suzuki (suzuki@hapis.k.u-tokyo.ac.jp)
 * -----
 * Copyright (c) 2019 Hapis Lab. All rights reserved.
 *
 */

#pragma once

#if WIN32
#include <codeanalysis\warnings.h>
#pragma warning(push)
#pragma warning(disable \
				: ALL_CODE_ANALYSIS_WARNINGS)
#endif
#include <Eigen/Core>
#include <Eigen/Geometry>
#if WIN32
#pragma warning(pop)
#endif

#include "geometry.hpp"
#include "gain.hpp"
#include "modulation.hpp"

using namespace std;

#if DLL_FOR_CAPI
using EtherCATAdapter = pair<string, string>;
using EtherCATAdapters = pair<string, string> *;
#else
using EtherCATAdapter = pair<shared_ptr<string>, shared_ptr<string>>;
using EtherCATAdapters = std::vector<EtherCATAdapter>;
#endif

namespace autd
{
class Controller
{
public:
	Controller() noexcept(false);
	~Controller() noexcept(false);

	bool isOpen();
	GeometryPtr geometry() noexcept;
	bool silentMode() noexcept;
	size_t remainingInBuffer();

	static EtherCATAdapters EnumerateAdapters(int &size);

	/*!
				 @brief Open device by link type and location.
					The scheme of location is as follows:
					ETHERCAT - <ams net id> or <ipv4 addr>:<ams net id> (ex. 192.168.1.2:192.168.1.3.1.1 ). The ipv4 addr will be extracted from leading 4 octets of ams net id if not specified.
					ETHERNET - ipv4 addr
					USB      - ignored
					SERIAL   - file discriptor
				 */
	void Open(LinkType type, std::string location = "");
	void SetSilentMode(bool silent) noexcept;
	void CalibrateModulation();
	void Close();

	void AppendGain(GainPtr gain);
	void AppendGainSync(GainPtr gain);
	void AppendModulation(ModulationPtr modulation);
	void AppendModulationSync(ModulationPtr modulation);
	void AppendSTMGain(GainPtr gain);
	void AppendSTMGain(const std::vector<GainPtr> &gain_list);
	void StartSTModulation(float freq);
	void StopSTModulation();
	void FinishSTModulation();
	void Flush();

	void LateralModulationAT(Eigen::Vector3f point, Eigen::Vector3f dir = Eigen::Vector3f::UnitY(), float lm_amp = 2.5, float lm_freq = 100);

	[[deprecated("AppendLateralGain is deprecated. Please use AppendSTMGain()")]] void AppendLateralGain(GainPtr gain);
	[[deprecated("AppendLateralGain is deprecated. Please use AppendSTMGain()")]] void AppendLateralGain(const std::vector<GainPtr> &gain_list);
	[[deprecated("StartLateralModulation is deprecated. Please use StartSTModulation()")]] void StartLateralModulation(float freq);
	[[deprecated("FinishLateralModulation is deprecated. Please use StopSTModulation()")]] void FinishLateralModulation();
	[[deprecated("ResetLateralGain is deprecated. Please use FinishSTModulation()")]] void ResetLateralGain();

private:
	class impl;
	std::unique_ptr<impl> _pimpl;
};
} // namespace autd