/*
 * File: libsoem.cpp
 * Project: linux
 * Created Date: 04/09/2019
 * Author: Shun Suzuki
 * -----
 * Last Modified: 04/09/2019
 * Modified By: Shun Suzuki (suzuki@hapis.k.u-tokyo.ac.jp)
 * -----
 * Copyright (c) 2019 Hapis Lab. All rights reserved.
 * 
 */

#include <stdio.h>
#include <signal.h>
#include <time.h>
#include <unistd.h>
#include <string.h>

#include <iostream>
#include <cstdint>
#include <mutex>
#include <memory>

#include "libsoem.hpp"
#include "ethercat.h"

using namespace std;

class libsoem::SOEMController::impl
{
public:
	void Open(const char *ifname, size_t devNum);
	void Send(size_t size, unique_ptr<uint8_t[]> buf);
	static void RTthread(union sigval sv);
	void Close();

private:
	void SetSendCheck(bool val);
	bool GetSendCheck();

	unique_ptr<uint8_t[]> _IOmap;
	bool _isOpened = false;
	size_t _devNum = 0;
	uint32_t _mmResult = 0;
	bool _sendCheck = false;
	mutex _mutex;
	timer_t _timer_id;
};

void libsoem::SOEMController::impl::Send(size_t size, unique_ptr<uint8_t[]> buf)
{
	if (_isOpened)
	{
		const auto header_size = MOD_SIZE + 4;
		const auto data_size = TRANS_NUM * 2;
		const auto includes_gain = ((size - header_size) / data_size) > 0;

		for (size_t i = 0; i < _devNum; i++)
		{
			if (includes_gain)
				memcpy(&_IOmap[OUTPUT_FRAME_SIZE * i], &buf[header_size + data_size * i], TRANS_NUM * 2);
			memcpy(&_IOmap[OUTPUT_FRAME_SIZE * i + TRANS_NUM * 2], &buf[0], MOD_SIZE + 4);
		}

		SetSendCheck(true);
		while (GetSendCheck())
			;
	}
}

void libsoem::SOEMController::impl::SetSendCheck(bool val)
{
	lock_guard<mutex> lock(_mutex);
	_sendCheck = val;
}

bool libsoem::SOEMController::impl::GetSendCheck()
{
	lock_guard<mutex> lock(_mutex);
	return _sendCheck;
}

void libsoem::SOEMController::impl::RTthread(union sigval sv)
{
	ec_send_processdata();
	(reinterpret_cast<SOEMController::impl *>(sv.sival_ptr))->SetSendCheck(false);
}

void libsoem::SOEMController::impl::Open(const char *ifname, size_t devNum)
{
	_devNum = devNum;
	auto size = (OUTPUT_FRAME_SIZE + INPUT_FRAME_SIZE) * _devNum;
	_IOmap = make_unique<uint8_t[]>(size);

	if (ec_init(ifname))
	{
		if (ec_config_init(0) > 0)
		{
			ec_config_map(&_IOmap[0]);
			ec_configdc();

			ec_statecheck(0, EC_STATE_SAFE_OP, EC_TIMEOUTSTATE * 4);

			ec_slave[0].state = EC_STATE_OPERATIONAL;
			ec_send_processdata();
			ec_receive_processdata(EC_TIMEOUTRET);

			struct itimerspec itval;
			struct sigevent se;

			itval.it_value.tv_sec = 0;
			itval.it_value.tv_nsec = 1000 * 1000;
			itval.it_interval.tv_sec = 0;
			itval.it_interval.tv_nsec = 1000 * 1000;

			memset(&se, 0, sizeof(se));
			se.sigev_value.sival_ptr = this;
			se.sigev_notify = SIGEV_THREAD;
			se.sigev_notify_function = RTthread;
			se.sigev_notify_attributes = NULL;

			if (timer_create(CLOCK_REALTIME, &se, &_timer_id) < 0)
			{
				cerr << "Error: timer_create." << endl;
			}

			if (timer_settime(_timer_id, 0, &itval, NULL) < 0)
			{
				cerr << "Error: timer_settime." << endl;
			}

			ec_writestate(0);

			auto chk = 200;
			do
			{
				ec_statecheck(0, EC_STATE_OPERATIONAL, 50000);
			} while (chk-- && (ec_slave[0].state != EC_STATE_OPERATIONAL));

			if (ec_slave[0].state == EC_STATE_OPERATIONAL)
			{
				_isOpened = true;
			}
			else
			{
				timer_delete(_timer_id);
				cerr << "One ore more slaves are not responding." << endl;
			}
		}
		else
		{
			cerr << "No slaves found!" << endl;
		}
	}
	else
	{
		cerr << "No socket connection on " << ifname << endl;
	}
}

void libsoem::SOEMController::impl::Close()
{
	if (_isOpened)
	{
		timer_delete(_timer_id);

		ec_slave[0].state = EC_STATE_INIT;
		ec_writestate(0);

		ec_close();

		_isOpened = false;
	}
}

libsoem::SOEMController::SOEMController()
{
	this->_pimpl = make_shared<impl>();
}

libsoem::SOEMController::~SOEMController()
{
	this->_pimpl->Close();
}

void libsoem::SOEMController::Open(const char *ifname, size_t devNum)
{
	this->_pimpl->Open(ifname, devNum);
}

void libsoem::SOEMController::Send(size_t size, unique_ptr<uint8_t[]> buf)
{
	this->_pimpl->Send(size, move(buf));
}

void libsoem::SOEMController::Close()
{
	this->_pimpl->Close();
}

vector<libsoem::EtherCATAdapterInfo> libsoem::EtherCATAdapterInfo::EnumerateAdapters()
{
	auto adapter = ec_find_adapters();
	auto _adapters = vector<EtherCATAdapterInfo>();
	while (adapter != NULL)
	{
		auto *info = new EtherCATAdapterInfo;
		info->desc = make_shared<string>(adapter->desc);
		info->name = make_shared<string>(adapter->name);
		_adapters.push_back(*info);
		adapter = adapter->next;
	}
	return _adapters;
}