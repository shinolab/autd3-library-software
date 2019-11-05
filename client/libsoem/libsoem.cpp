/*
 * File: libsoem.cpp
 * Project: win32
 * Created Date: 23/08/2019
 * Author: Shun Suzuki
 * -----
 * Last Modified: 05/11/2019
 * Modified By: Shun Suzuki (suzuki@hapis.k.u-tokyo.ac.jp)
 * -----
 * Copyright (c) 2019 Hapis Lab. All rights reserved.
 *
 */

#if (_WIN32 || _WIN64)
#define WINDOWS
#elif defined __APPLE__
#define MACOSX
#elif defined __linux__
#define LINUX
#else
#error "Not supported."
#endif

#ifdef WINDOWS
#define __STDC_LIMIT_MACROS
#include <WinError.h>
#else
#include <stdio.h>
#include <signal.h>
#include <time.h>
#include <unistd.h>
#include <string.h>
#endif

#ifdef MACOSX
#include <dispatch/dispatch.h>
#endif

#include <iostream>
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
	void Open(const char* ifname, size_t devNum, uint32_t ec_sm3_cyctime_ns, uint32_t ec_sync0_cyctime_ns, size_t header_size, size_t body_size, size_t input_frame_size);
	void Send(size_t size, unique_ptr<uint8_t[]> buf);
	vector<uint16_t> Read(size_t input_frame_idx);
	void Close();

	bool _isOpened = false;

private:
#ifdef WINDOWS
	static void CALLBACK RTthread(PVOID lpParam, BOOLEAN TimerOrWaitFired);
#elif defined MACOSX
	static void RTthread(SOEMController::impl* pimpl);
#elif defined LINUX
	static void RTthread(union sigval sv);
#endif
	void SetupSync0(bool actiavte, uint32_t CycleTime);
	void CreateCopyThread(size_t header_size, size_t body_size);

	vector<uint8_t> _IOmap;
	uint32_t _sync0_cyctime = 0;

	queue<size_t> _send_size_q;
	queue<unique_ptr<uint8_t[]>> _send_buf_q;
	thread _cpy_thread;
	condition_variable _cpy_cond;
	condition_variable _send_cond;
	bool _sent = false;
	mutex _cpy_mtx;
	mutex _send_mtx;

	size_t _devNum = 0;

#ifdef WINDOWS
	HANDLE _timerQueue = NULL;
	HANDLE _timer = NULL;
#elif defined MACOSX
	dispatch_queue_t _queue;
	dispatch_source_t _timer;
#elif defined LINUX
	timer_t _timer_id;
#endif
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

#ifdef WINDOWS
void CALLBACK SOEMController::impl::RTthread(PVOID lpParam, BOOLEAN TimerOrWaitFired)
{
	const auto pimpl = (reinterpret_cast<SOEMController::impl*>(lpParam));
#elif defined MACOSX
void libsoem::SOEMController::impl::RTthread(SOEMController::impl * pimpl)
{
#elif defined LINUX
void libsoem::SOEMController::impl::RTthread(union sigval sv)
{
	const auto pimpl = (reinterpret_cast<SOEMController::impl*>(sv.sival_ptr));
#endif
	ec_send_processdata();
	{
		lock_guard<mutex> lk(pimpl->_send_mtx);
		pimpl->_sent = true;
	}
	pimpl->_send_cond.notify_all();
	ec_receive_processdata(EC_TIMEOUTRET);
}

vector<uint16_t> SOEMController::impl::Read(size_t input_frame_idx)
{
	vector<uint16_t> res;
	for (size_t i = 0; i < _devNum; i++)
	{
		uint16_t base = ((uint16_t)_IOmap[input_frame_idx + 2 * i + 1] << 8) | _IOmap[input_frame_idx + 2 * i];
		res.push_back(base);
	}
	return res;
}

void SOEMController::impl::SetupSync0(bool actiavte, uint32_t CycleTime)
{
	auto exceed = CycleTime > 1000000u;
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

void SOEMController::impl::CreateCopyThread(size_t header_size, size_t body_size)
{
	_cpy_thread = thread([this](size_t header_size, size_t body_size) {
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
				const auto includes_gain = ((size - header_size) / body_size) > 0;
				const auto output_frame_size = header_size + body_size;

				for (size_t i = 0; i < _devNum; i++)
				{
					if (includes_gain)
						memcpy(&_IOmap[output_frame_size * i], &buf[header_size + body_size * i], body_size);
					memcpy(&_IOmap[output_frame_size * i + body_size], &buf[0], header_size);
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
		},
		header_size, body_size);
}

void SOEMController::impl::Open(const char* ifname, size_t devNum, uint32_t ec_sm3_cyctime_ns, uint32_t ec_sync0_cyctime_ns, size_t header_size, size_t body_size, size_t input_frame_size)
{
	_devNum = devNum;
	auto size = (header_size + body_size + input_frame_size) * _devNum;
	if (_IOmap.size() != size)
	{
		_IOmap = vector<uint8_t>(size, 0);
	}

	_sync0_cyctime = ec_sync0_cyctime_ns;

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

			ec_writestate(0);

			auto chk = 200;
			do
			{
				ec_statecheck(0, EC_STATE_OPERATIONAL, 50000);
			} while (chk-- && (ec_slave[0].state != EC_STATE_OPERATIONAL));

			if (ec_slave[0].state == EC_STATE_OPERATIONAL)
			{
				_isOpened = true;

				SetupSync0(true, _sync0_cyctime);

#ifdef WINDOWS
				_timerQueue = CreateTimerQueue();
				if (_timerQueue == NULL)
					cerr << "CreateTimerQueue failed." << endl;

				if (!CreateTimerQueueTimer(&_timer, _timerQueue, (WAITORTIMERCALLBACK)RTthread, reinterpret_cast<void*>(this), 0, ec_sm3_cyctime_ns / 1000 / 1000, 0))
					cerr << "CreateTimerQueueTimer failed." << endl;
#elif defined MACOSX
				_queue = dispatch_queue_create("timerQueue", 0);

				_timer = dispatch_source_create(DISPATCH_SOURCE_TYPE_TIMER, 0, 0, _queue);
				dispatch_source_set_event_handler(_timer, ^ {
				  RTthread(this);
					});

				dispatch_source_set_cancel_handler(_timer, ^ {
				  dispatch_release(_timer);
				  dispatch_release(_queue);
					});

				dispatch_time_t start = dispatch_time(DISPATCH_TIME_NOW, 0);
				dispatch_source_set_timer(_timer, start, ec_sm3_cyctime_ns, 0);
				dispatch_resume(_timer);
#elif defined LINUX
				struct itimerspec itval;
				struct sigevent se;

				itval.it_value.tv_sec = 0;
				itval.it_value.tv_nsec = ec_sm3_cyctime_ns;
				itval.it_interval.tv_sec = 0;
				itval.it_interval.tv_nsec = ec_sm3_cyctime_ns;

				memset(&se, 0, sizeof(se));
				se.sigev_value.sival_ptr = this;
				se.sigev_notify = SIGEV_THREAD;
				se.sigev_notify_function = RTthread;
				se.sigev_notify_attributes = NULL;

				if (timer_create(CLOCK_REALTIME, &se, &_timer_id) < 0)
					cerr << "Error: timer_create." << endl;

				if (timer_settime(_timer_id, 0, &itval, NULL) < 0)
					cerr << "Error: timer_settime." << endl;
#endif

				CreateCopyThread(header_size, body_size);
			}
			else
			{
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

		fill(_IOmap.begin(), _IOmap.end(), 0x0000);
		this_thread::sleep_for(chrono::milliseconds(200));

#ifdef WINDOWS
		if (!DeleteTimerQueueTimer(_timerQueue, _timer, 0))
		{
			if (GetLastError() != ERROR_IO_PENDING)
				cerr << "DeleteTimerQueue failed." << endl;
			else
				Sleep(200);
		}
#elif defined MACOSX
		dispatch_source_cancel(_timer);
#elif defined LINUX
		timer_delete(_timer_id);
#endif
		SetupSync0(false, _sync0_cyctime);

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

void SOEMController::Open(const char* ifname, size_t devNum, uint32_t ec_sm3_cyctime_ns, uint32_t ec_sync0_cyctime_ns, size_t header_size, size_t body_size, size_t input_frame_size)
{
	this->_pimpl->Open(ifname, devNum, ec_sm3_cyctime_ns, ec_sync0_cyctime_ns, header_size, body_size, input_frame_size);
}

void SOEMController::Send(size_t size, unique_ptr<uint8_t[]> buf)
{
	this->_pimpl->Send(size, move(buf));
}

vector<uint16_t> SOEMController::Read(size_t input_frame_idx)
{
	return this->_pimpl->Read(input_frame_idx);
}

void SOEMController::Close()
{
	this->_pimpl->Close();
}

bool SOEMController::isOpen()
{
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
