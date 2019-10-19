/*
 * File: libsoem.cpp
 * Project: win32
 * Created Date: 23/08/2019
 * Author: Shun Suzuki
 * -----
 * Last Modified: 19/10/2019
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
#include <condition_variable>
#include <memory>
#include <queue>
#include <vector>
#include <thread>

#include "libsoem.hpp"
#include "ethercat.h"

using namespace libsoem;
using namespace std;

class SOEMController::impl
{
public:
	void Open(const char* ifname, size_t devNum, uint32_t CycleTime);
	void Send(size_t size, unique_ptr<uint8_t[]> buf);
	void Close();

	bool _isOpened = false;

private:
	static void CALLBACK RTthread(PVOID lpParam, BOOLEAN TimerOrWaitFired);
	void SetupSync0(bool actiavte, uint32_t CycleTime);
	void CreateCopyThread();

	vector<uint8_t> _IOmap;
	uint32_t _cycleTime = 0;

	queue<size_t> _send_size_q;
	queue<unique_ptr<uint8_t[]>> _send_buf_q;
	thread _cpy_thread;
	condition_variable _cpy_cond;
	condition_variable _send_cond;
	bool _sent = false;
	mutex _cpy_mtx;
	mutex _send_mtx;

	size_t _devNum = 0;
	HANDLE _timerQueue = NULL;
	HANDLE _timer = NULL;
};

void SOEMController::impl::Send(size_t size, unique_ptr<uint8_t[]> buf)
{
	{
		unique_lock<mutex> lk(_cpy_mtx);
		_send_size_q.push(size);
		_send_buf_q.push(move(buf));
	}
	_cpy_cond.notify_all();
}

void CALLBACK SOEMController::impl::RTthread(PVOID lpParam, BOOLEAN TimerOrWaitFired)
{
	ec_send_processdata();
	const auto impl = (reinterpret_cast<SOEMController::impl*>(lpParam));
	{
		lock_guard<mutex> lk(impl->_send_mtx);
		impl->_sent = true;
	}
	impl->_send_cond.notify_all();
}

void SOEMController::impl::SetupSync0(bool actiavte, uint32_t CycleTime)
{
	auto exceed = static_cast<unsigned long>(_devNum - 1)* static_cast<unsigned long>(CycleTime) > 0x7ffffffful;
	for (uint16 slave = 1; slave <= _devNum; slave++)
	{
		if (exceed)
		{
			ec_dcsync0(slave, actiavte, CycleTime, 0); // SYNC0
		}
		else
		{
			int shift = static_cast<int>(_devNum) - slave;
			ec_dcsync0(slave, actiavte, CycleTime, shift * CycleTime); // SYNC0
		}
	}
}

void SOEMController::impl::CreateCopyThread()
{
	_cpy_thread = thread([&] {
		while (_isOpened)
		{
			unique_ptr<uint8_t[]> buf = nullptr;
			size_t size = 0;
			{
				unique_lock<mutex> lk(_cpy_mtx);
				_cpy_cond.wait(lk, [&] {
					return _send_buf_q.size() > 0 || !_isOpened;
					});
				if (_send_buf_q.size() > 0)
				{
					buf = move(_send_buf_q.front());
					size = _send_size_q.front();
				}
			}

			if (buf != nullptr && _isOpened)
			{
				const auto header_size = MOD_FRAME_SIZE + 4;
				const auto data_size = TRANS_NUM * 2;
				const auto includes_gain = ((size - header_size) / data_size) > 0;

				for (size_t i = 0; i < _devNum; i++)
				{
					if (includes_gain)
						memcpy(&_IOmap[OUTPUT_FRAME_SIZE * i], &buf[header_size + data_size * i], data_size);
					memcpy(&_IOmap[OUTPUT_FRAME_SIZE * i + data_size], &buf[0], header_size);
				}

				{
					unique_lock<mutex> lk(_send_mtx);
					_sent = false;
					_send_cond.wait(lk, [this] { return _sent || !_isOpened; });
				}

				_send_size_q.pop();
				_send_buf_q.pop();
			}
		}
		});
}

void SOEMController::impl::Open(const char* ifname, size_t devNum, uint32_t CycleTime)
{
	_devNum = devNum;
	auto size = (OUTPUT_FRAME_SIZE + INPUT_FRAME_SIZE) * _devNum;
	if (_IOmap.size() != size) {
		_IOmap = vector<uint8_t>(size, 0);
	}

	_cycleTime = CycleTime;

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

			if (!CreateTimerQueueTimer(&_timer, _timerQueue, (WAITORTIMERCALLBACK)RTthread, reinterpret_cast<void*>(this), 0, SM3_CYCLE_TIME_MILLI_SEC, 0))
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

				SetupSync0(true, _cycleTime);

				CreateCopyThread();
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
		_isOpened = false;

		_cpy_cond.notify_all();
		_send_cond.notify_all();
		if (this_thread::get_id() != _cpy_thread.get_id() && this->_cpy_thread.joinable())
			this->_cpy_thread.join();

		if (!DeleteTimerQueueTimer(_timerQueue, _timer, 0))
		{
			if (GetLastError() != ERROR_IO_PENDING)
				cerr << "DeleteTimerQueue failed." << endl;
			else
				Sleep(200);
		}

		SetupSync0(false, _cycleTime);

		auto size = OUTPUT_FRAME_SIZE * _devNum;
		memset(&_IOmap[0], 0x00, size);
		for (int i = 0; i < 200; i++)
		{
			ec_send_processdata();
		}

		ec_slave[0].state = EC_STATE_INIT;
		ec_writestate(0);

		ec_statecheck(0, EC_STATE_INIT, EC_TIMEOUTSTATE);

		ec_close();
	}
}

SOEMController::SOEMController()
{
	this->_pimpl = make_unique<impl>();
}

SOEMController::~SOEMController()
{
	this->_pimpl->Close();
}

void SOEMController::Open(const char* ifname, size_t devNum, uint32_t CycleTime)
{
	this->_pimpl->Open(ifname, devNum, CycleTime);
}

void SOEMController::Send(size_t size, unique_ptr<uint8_t[]> buf)
{
	this->_pimpl->Send(size, move(buf));
}

void SOEMController::Close()
{
	this->_pimpl->Close();
}

bool SOEMController::isOpen() {
	return this->_pimpl->_isOpened;
}

vector<EtherCATAdapterInfo> EtherCATAdapterInfo::EnumerateAdapters()
{
	auto adapter = ec_find_adapters();
	auto _adapters = vector<EtherCATAdapterInfo>();
	while (adapter != NULL)
	{
		auto* info = new EtherCATAdapterInfo;
		info->desc = make_shared<string>(adapter->desc);
		info->name = make_shared<string>(adapter->name);
		_adapters.push_back(*info);
		adapter = adapter->next;
	}
	return _adapters;
}
