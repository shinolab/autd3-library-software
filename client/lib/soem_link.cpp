/*
 * File: soem_link.cpp
 * Project: lib
 * Created Date: 24/08/2019
 * Author: Shun Suzuki
 * -----
 * Last Modified: 20/10/2019
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
	_isOpen = _cnt->isOpen();
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

void autd::internal::SOEMLink::CalibrateModulation()
{
	cout << "Start calibrating modulation..." << endl;
	constexpr auto MOD_PERIOD_MS = (uint32_t)((MOD_BUF_SIZE / MOD_SAMPLING_FREQ) * 1000);

	cout << "DEV No.\t| Header\t| BASE" << endl;
	auto v = _cnt->Read();
	cout << "************* PRE ******************" << endl;
	for (size_t i = 0; i < v.size(); i++)
	{
		auto h = (v.at(i) & 0xC000) >> 14;
		auto base = v.at(i) & 0x3FFF;
		cout << i << "\t| " << (int)h << "\t\t| " << (int)base << endl;
	}

	_cnt->Close();
	_cnt->Open(_ifname.c_str(), _devNum, MOD_PERIOD_MS * 1000 * 1000);

	auto size = sizeof(RxGlobalHeader);
	auto body = make_unique<uint8_t[]>(size);
	auto* header = reinterpret_cast<RxGlobalHeader*>(&body[0]);
	header->msg_id = 0xFF;

	Send(size, move(body));

	std::this_thread::sleep_for(std::chrono::milliseconds((_devNum + 5) * MOD_PERIOD_MS));

	_cnt->Close();
	_cnt->Open(_ifname.c_str(), _devNum);

	std::this_thread::sleep_for(std::chrono::milliseconds(1000));

	v = _cnt->Read();
	cout << "************* AFTER ******************" << endl;
	for (size_t i = 0; i < v.size(); i++)
	{
		auto h = (v.at(i) & 0xC000) >> 14;
		auto base = v.at(i) & 0x3FFF;
		cout << i << "\t| " << (int)h << "\t\t| " << (int)base << endl;
	}

	cout << "Finish calibrating modulation..." << endl;
}
