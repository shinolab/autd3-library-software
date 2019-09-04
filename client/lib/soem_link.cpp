/*
*  soem_link.cpp
*  autd3
*
*  Created by Shun Suzuki on 08/23/2019.
*  Copyright © 2019 Hapis Lab. All rights reserved.
*
*/

#include "libsoem.hpp"
#include "soem_link.hpp"

#include <vector>

using namespace std;

void autd::internal::SOEMLink::Open(std::string ifname) {
	_cnt = std::make_unique<libsoem::SOEMController>();

	auto ifname_and_devNum = split(ifname, ':');
	_cnt->Open(ifname_and_devNum[0].c_str(), stoi(ifname_and_devNum[1]));
	_isOpen = true;
}

void autd::internal::SOEMLink::Close() {
	if (_isOpen) {
		_cnt->Close();
		_isOpen = false;
	}
}

void autd::internal::SOEMLink::Send(size_t size, std::unique_ptr<uint8_t[]> buf) {
	if (_isOpen) {
		_cnt->Send(size, std::move(buf));
	}
}

bool autd::internal::SOEMLink::isOpen() {
	return _isOpen;
}