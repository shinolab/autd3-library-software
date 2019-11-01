/*
 * File: soem_link.cpp
 * Project: lib
 * Created Date: 24/08/2019
 * Author: Shun Suzuki
 * -----
 * Last Modified: 01/11/2019
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

// DEBUG
#include <fstream>

using namespace std;

void autd::internal::SOEMLink::Open(std::string ifname)
{
	_cnt = std::make_unique<libsoem::SOEMController>();

	auto ifname_and_devNum = split(ifname, ':');
	_devNum = stoi(ifname_and_devNum[1]);
	_ifname = ifname_and_devNum[0];
	_cnt->Open(_ifname.c_str(), _devNum, EC_SM3_CYCLE_TIME_NANO_SEC, EC_SYNC0_CYCLE_TIME_NANO_SEC, HEADER_SIZE, NUM_TRANS_IN_UNIT * 2, EC_INPUT_FRAME_SIZE);
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
	auto log = [](ofstream &file, const vector<uint16_t> &v) {
		for (size_t i = 0; i < v.size(); i++)
		{
			auto h = (v.at(i) & 0xC000) >> 14;
			auto base = v.at(i) & 0x3FFF;
			file << i << "\t| " << (int)h << "\t\t| " << (int)base << endl;
		}
	};

	auto succeed_calib = [](const vector<uint16_t> &v) {
		auto min = *min_element(v.begin(), v.end());
		for (size_t i = 0; i < v.size(); i++)
		{
			auto h = (v.at(i) & 0xC000) >> 14;
			auto base = v.at(i) & 0x3FFF;
			if (h == 3 || (base - min) % 4 != 0)
				return false;
		}

		return true;
	};

	cout << "Start calibrating modulation..." << endl;
	constexpr auto MOD_PERIOD_MS = (uint32_t)((MOD_BUF_SIZE / MOD_SAMPLING_FREQ) * 1000);

	ofstream full_log_file("full_log.txt");
	ofstream error_log_file("error_log.txt");

	full_log_file << "DEV No.\t| Header\t| BASE" << endl;
	error_log_file << "DEV No.\t| Header\t| BASE" << endl;

	for (size_t i = 0; i < 20000; i++)
	{
		cout << i << "-th test start..." << endl;
		full_log_file << i << "-th test start..." << endl;

		_cnt->Close();
		_cnt->Open(_ifname.c_str(), _devNum, EC_SM3_CYCLE_TIME_NANO_SEC, MOD_PERIOD_MS * 1000 * 1000, HEADER_SIZE, NUM_TRANS_IN_UNIT * 2, EC_INPUT_FRAME_SIZE);

		auto size = sizeof(RxGlobalHeader);
		auto body = make_unique<uint8_t[]>(size);
		auto *header = reinterpret_cast<RxGlobalHeader *>(&body[0]);
		header->msg_id = 0xFF;

		Send(size, move(body));

		std::this_thread::sleep_for(std::chrono::milliseconds((_devNum + 1) * MOD_PERIOD_MS));

		_cnt->Close();
		_cnt->Open(_ifname.c_str(), _devNum, EC_SM3_CYCLE_TIME_NANO_SEC, EC_SYNC0_CYCLE_TIME_NANO_SEC, HEADER_SIZE, NUM_TRANS_IN_UNIT * 2, EC_INPUT_FRAME_SIZE);

		std::this_thread::sleep_for(std::chrono::milliseconds(1000));

		auto v = _cnt->Read(EC_OUTPUT_FRAME_SIZE * _devNum);
		log(full_log_file, v);
		if (!succeed_calib(v))
		{
			cout << "failed!" << endl;
			log(error_log_file, v);
		}
	}

	full_log_file << "finish" << endl;
	error_log_file << "finish" << endl;

	full_log_file.close();
	error_log_file.close();

	cout << "Finish calibrating modulation..." << endl;
}
