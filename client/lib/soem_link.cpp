/*
 * File: soem_link.cpp
 * Project: lib
 * Created Date: 24/08/2019
 * Author: Shun Suzuki
 * -----
 * Last Modified: 14/10/2019
 * Modified By: Shun Suzuki (suzuki@hapis.k.u-tokyo.ac.jp)
 * -----
 * Copyright (c) 2019 Hapis Lab. All rights reserved.
 *
 */

#include "libsoem.hpp"
#include "soem_link.hpp"
#include "privdef.hpp"

#include <vector>
#include <thread>
#include <chrono>
#include <iostream>

using namespace std;

void autd::internal::SOEMLink::Open(std::string ifname)
{
	_cnt = std::make_unique<libsoem::SOEMController>();

	auto ifname_and_devNum = split(ifname, ':');
	_devNum = stoi(ifname_and_devNum[1]);
	_ifname = ifname_and_devNum[0];
	_cnt->Open(_ifname.c_str(), _devNum);
	_isOpen = true;
}

void autd::internal::SOEMLink::Close()
{
	if (_isOpen)
	{
		_cnt->Close();
		_isOpen = false;
	}
}

void autd::internal::SOEMLink::Send(size_t size, std::unique_ptr<uint8_t[]> buf)
{
	if (_isOpen)
	{
		_cnt->Send(size, std::move(buf));
	}
}

bool autd::internal::SOEMLink::isOpen()
{
	return _isOpen;
}
