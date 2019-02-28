//
//  groupedgain.cpp
//
//  Created by Shun Suzuki on 09 / 07 / 2018.
//
//


#include <stdio.h>
#include <map>
#include <codeanalysis\warnings.h>
#pragma warning( push )
#pragma warning ( disable : ALL_CODE_ANALYSIS_WARNINGS )
#include <boost/assert.hpp>
#pragma warning( pop )
#include "autd3.hpp"
#include "privdef.hpp"

autd::GainPtr autd::GroupedGain::Create(std::map<int, autd::GainPtr> gainmap) {
	auto gain =CreateHelper<GroupedGain>();
	gain->_gainmap = gainmap;
	gain->_geometry = GeometryPtr(nullptr);
	return gain;
}

void autd::GroupedGain::build() {
	if (this->built()) return;
	if (this->geometry() == nullptr) BOOST_ASSERT_MSG(false, "Geometry is required to build Gain");

	this->_data.clear();

	const int ndevice = this->geometry()->numDevices();
	for (int i = 0; i < ndevice; i++) {
		this->_data[this->geometry()->deviceIdForDeviceIdx(i)].resize(NUM_TRANS_IN_UNIT);
	}

	for (std::pair<int, GainPtr> p : this->_gainmap) {
		auto g = p.second;
		g->SetGeometry(this->geometry());
		g->build();
	}

	for (int i = 0; i < ndevice; i++)
	{
		auto groupId = this->geometry()->GroupIDForDeviceID(i);
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

