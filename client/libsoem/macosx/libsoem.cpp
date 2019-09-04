/*
 * File: libsoem.cpp
 * Project: macosx
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

#include <dispatch/dispatch.h>

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
	static void RTthread(SOEMController::impl *ptr);
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
	dispatch_queue_t _queue;
	dispatch_source_t _timer;
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

void libsoem::SOEMController::impl::RTthread(SOEMController::impl *ptr)
{
	ec_send_processdata();
	ptr->SetSendCheck(false);
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

			_queue = dispatch_queue_create("timerQueue", 0);

			_timer = dispatch_source_create(DISPATCH_SOURCE_TYPE_TIMER, 0, 0, _queue);
			dispatch_source_set_event_handler(_timer, ^{
			  RTthread(this);
			});

			dispatch_source_set_cancel_handler(_timer, ^{
			  dispatch_release(_timer);
			  dispatch_release(_queue);
			});

			dispatch_time_t start = dispatch_time(DISPATCH_TIME_NOW, 0);
			dispatch_source_set_timer(_timer, start, 1000 * 1000, 0);
			dispatch_resume(_timer);

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
				dispatch_source_cancel(_timer);
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
		dispatch_source_cancel(_timer);

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