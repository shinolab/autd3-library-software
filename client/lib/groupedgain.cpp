/*
*  groupedgain.cpp
*  autd3
*
*  Created by Shun Suzuki on 09/07/18.
*  Copyright © 2018-2019 Hapis Lab. All rights reserved.
*
*/

#include <stdio.h>
#include <map>

#if WIN32
#include <codeanalysis\warnings.h>
#pragma warning(push)
#pragma warning(disable:ALL_CODE_ANALYSIS_WARNINGS)
#endif
#include <boost/assert.hpp>
#if WIN32
#pragma warning(pop)
#endif

#include "autd3.hpp"
#include "controller.hpp"
#include "gain.hpp"
#include "privdef.hpp"

using namespace autd;

GainPtr GroupedGain::Create(std::map<int, GainPtr> gainmap) {
	auto gain = CreateHelper<GroupedGain>();
	gain->_gainmap = gainmap;
	gain->_geometry = nullptr;
	return gain;
}

void GroupedGain::build() {
	if (this->built()) return;
	auto geo = this->geometry();
	if (geo == nullptr) BOOST_ASSERT_MSG(false, "Geometry is required to build Gain");

	this->_data.clear();

	const auto ndevice = geo->numDevices();
	for (int i = 0; i < ndevice; i++) {
		this->_data[geo->deviceIdForDeviceIdx(i)].resize(NUM_TRANS_IN_UNIT);
	}

	for (std::pair<int, GainPtr> p : this->_gainmap) {
		auto g = p.second;
		g->SetGeometry(geo);
		g->build();
	}

	for (int i = 0; i < ndevice; i++)
	{
		auto groupId = geo->GroupIDForDeviceID(i);
		if (_gainmap.count(groupId)) {
			auto data = _gainmap[groupId]->data();
			this->_data[i] = data[i];
		}
		else {
			this->_data[i] = std::vector<uint16_t>(NUM_TRANS_IN_UNIT, 0x0000);
		}
	}

	this->_built = true;
}

