/*
 * File: libsoem.cpp
 * Project: win32
 * Created Date: 23/08/2019
 * Author: Shun Suzuki
 * -----
 * Last Modified: 24/09/2019
 * Modified By: Shun Suzuki (suzuki@hapis.k.u-tokyo.ac.jp)
 * -----
 * Copyright (c) 2019 Hapis Lab. All rights reserved.
 *
 */

#define __STDC_LIMIT_MACROS

#include <iostream>
#include <WinError.h>
#include <cstdint>
#include <mutex>
#include <memory>
#include <queue>

#include "libsoem.hpp"
#include "ethercat.h"

using namespace libsoem;
using namespace std;

class SOEMController::impl
{
public:
	void Open(const char *ifname, size_t devNum);
	void Send(size_t size, unique_ptr<uint8_t[]> buf);
	static void CALLBACK RTthread(PVOID lpParam, BOOLEAN TimerOrWaitFired);
	void Close();

private:
	//void SetSendCheck(bool val);
	//bool GetSendCheck();

	unique_ptr<uint8_t[]> _IOmap;
	queue<size_t> _send_size_q;
	queue<unique_ptr<uint8_t[]>> _send_buf_q;
	bool _isOpened = false;
	size_t _devNum = 0;
	uint32_t _mmResult = 0;
	HANDLE _timerQueue = NULL;
	HANDLE _timer = NULL;
};

void SOEMController::impl::Send(size_t size, unique_ptr<uint8_t[]> buf)
{
	_send_size_q.push(size);
	_send_buf_q.push(move(buf));
}

// void SOEMController::impl::SetSendCheck(bool val)
// {
// 	lock_guard<mutex> lock(_mutex);
// 	_sendCheck = val;
// }

// bool SOEMController::impl::GetSendCheck()
// {
// 	lock_guard<mutex> lock(_mutex);
// 	return _sendCheck;
// }

void CALLBACK SOEMController::impl::RTthread(PVOID lpParam, BOOLEAN TimerOrWaitFired)
{
	const auto impl = (reinterpret_cast<SOEMController::impl *>(lpParam));
	if (impl->_isOpened)
	{
		if (impl->_send_buf_q.size() > 0)
		{
			const auto size = impl->_send_size_q.front();
			const auto buf = move(impl->_send_buf_q.front());

			const auto header_size = MOD_SIZE + 4;
			const auto data_size = TRANS_NUM * 2;
			const auto includes_gain = ((size - header_size) / data_size) > 0;

			for (size_t i = 0; i < impl->_devNum; i++)
			{
				if (includes_gain)
					memcpy(&impl->_IOmap[OUTPUT_FRAME_SIZE * i], &buf[header_size + data_size * i], TRANS_NUM * 2);
				memcpy(&impl->_IOmap[OUTPUT_FRAME_SIZE * i + TRANS_NUM * 2], &buf[0], MOD_SIZE + 4);
			}
		}
		ec_send_processdata();

		if (impl->_send_size_q.size() > 0)
		{
			impl->_send_size_q.pop();
			impl->_send_buf_q.pop();
		}
	}

	// ec_send_processdata();
	// (reinterpret_cast<SOEMController::impl *>(lpParam))->SetSendCheck(false);
}

void SOEMController::impl::Open(const char *ifname, size_t devNum)
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

			_timerQueue = CreateTimerQueue();
			if (_timerQueue == NULL)
				cerr << "CreateTimerQueue failed." << endl;

			if (!CreateTimerQueueTimer(&_timer, _timerQueue, (WAITORTIMERCALLBACK)RTthread, reinterpret_cast<void *>(this), 0, 1, 0))
				cerr << "CreateTimerQueueTimer failed." << endl;

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
				if (!DeleteTimerQueueTimer(_timerQueue, _timer, 0))
					cerr << "DeleteTimerQueue failed." << endl;
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

void SOEMController::impl::Close()
{
	if (_isOpened)
	{
		while (_send_buf_q.size() > 0)
		{
			Sleep(1);
		}
		Sleep(1);

		if (!DeleteTimerQueueTimer(_timerQueue, _timer, 0))
		{
			if (GetLastError() != ERROR_IO_PENDING)
				cerr << "DeleteTimerQueue failed." << endl;
		}

		auto size = (OUTPUT_FRAME_SIZE + INPUT_FRAME_SIZE) * _devNum;
		vector<uint8_t> null_vec(size,0x00);
		memcpy(&_IOmap[0], &null_vec[0], size);
		ec_send_processdata();

		ec_slave[0].state = EC_STATE_INIT;
		ec_writestate(0);

		ec_close();

		_isOpened = false;
	}
}

SOEMController::SOEMController()
{
	this->_pimpl = make_shared<impl>();
}

SOEMController::~SOEMController()
{
	this->_pimpl->Close();
}

void SOEMController::Open(const char *ifname, size_t devNum)
{
	this->_pimpl->Open(ifname, devNum);
}

void SOEMController::Send(size_t size, unique_ptr<uint8_t[]> buf)
{
	this->_pimpl->Send(size, move(buf));
}

void SOEMController::Close()
{
	this->_pimpl->Close();
}

vector<EtherCATAdapterInfo> EtherCATAdapterInfo::EnumerateAdapters()
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