/*
 * File: controller.hpp
 * Project: include
 * Created Date: 11/04/2018
 * Author: Shun Suzuki
 * -----
 * Last Modified: 11/10/2019
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
	/*!
		 @brief Open device by link type and location.
			The scheme of location is as follows:
			ETHERCAT - <ams net id> or <ipv4 addr>:<ams net id> (ex. 192.168.1.2:192.168.1.3.1.1 ). The ipv4 addr will be extracted from leading 4 octets of ams net id if not specified.
			ETHERNET - ipv4 addr
			USB      - ignored
			SERIAL   - file discriptor
		 */
	void Open(LinkType type, std::string location = "");
	bool isOpen();
	void Close();

	static EtherCATAdapters EnumerateAdapters(int &size);

	size_t remainingInBuffer();
	GeometryPtr geometry() noexcept;
	void SetGeometry(const GeometryPtr &geometry) noexcept;

	void SetSilentMode(bool silent) noexcept;
	bool silentMode() noexcept;
	void AppendGain(GainPtr gain);
	void AppendGainSync(GainPtr gain);
	void AppendModulation(ModulationPtr modulation);
	void AppendModulationSync(ModulationPtr modulation);
	void AppendLateralGain(GainPtr gain);
	void AppendLateralGain(const std::vector<GainPtr> &gain_list);
	void StartLateralModulation(float freq);
	void FinishLateralModulation();
	void ResetLateralGain();
	void Flush();
	void CalibrateModulation();

	void LateralModulationAT(Eigen::Vector3f point, Eigen::Vector3f dir = Eigen::Vector3f::UnitY(), float lm_amp = 2.5, float lm_freq = 100);

private:
	class impl;
	class lateraltimer;
	std::shared_ptr<impl> _pimpl;
	std::unique_ptr<lateraltimer> _ptimer;
};
} // namespace autd