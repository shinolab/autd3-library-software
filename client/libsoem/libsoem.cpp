#define __STDC_LIMIT_MACROS

#include <iostream>
#include <cstdint>
#include <mutex>
#include <memory>

#include "libsoem.hpp"
#include "ethercat.h"

using namespace libsoem;

class SOEMController::impl
{
public:
	void Open(const char* ifname, size_t devNum);
	void Send(size_t size, std::unique_ptr<uint8_t[]> buf);
	static void CALLBACK RTthread(PVOID lpParam, BOOLEAN TimerOrWaitFired);
	void Close();

private:
	void SetSendCheck(bool val);
	bool GetSendCheck();

	std::unique_ptr<uint8_t[]>  _IOmap;
	bool _isClosed;
	size_t _devNum;
	uint32_t _mmResult;
	bool _sendCheck;
	std::mutex _mutex;
	HANDLE _timerQueue;
	HANDLE _timer;
};

void SOEMController::impl::Send(size_t size, std::unique_ptr<uint8_t[]> buf)
{
	if (!_isClosed) {
		const auto header_size = MOD_SIZE + 4;
		const auto data_size = TRANS_NUM * 2;
		const auto includes_gain = ((size - header_size) / data_size) > 0;

		for (size_t i = 0; i < _devNum; i++)
		{
			if (includes_gain) memcpy(&_IOmap[OUTPUT_FRAME_SIZE * i], &buf[header_size + data_size * i], TRANS_NUM * 2);
			memcpy(&_IOmap[OUTPUT_FRAME_SIZE * i + TRANS_NUM * 2], &buf[0], MOD_SIZE + 4);
		}
		SetSendCheck(true);
		while (GetSendCheck());
	}
}

void SOEMController::impl::SetSendCheck(bool val) {
	std::lock_guard<std::mutex> lock(_mutex);
	_sendCheck = val;
}

bool SOEMController::impl::GetSendCheck() {
	std::lock_guard<std::mutex> lock(_mutex);
	return _sendCheck;
}

void CALLBACK SOEMController::impl::RTthread(PVOID lpParam, BOOLEAN TimerOrWaitFired) {
	ec_send_processdata();
	(reinterpret_cast<SOEMController::impl*>(lpParam))->SetSendCheck(false);
}

void SOEMController::impl::Open(const char* ifname, size_t devNum)
{
	_devNum = devNum;
	auto size = (OUTPUT_FRAME_SIZE + INPUT_FRAME_SIZE) * _devNum;
	_IOmap = std::make_unique<uint8_t[]>(size);

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
				std::cerr << "CreateTimerQueue failed." << std::endl;

			if (!CreateTimerQueueTimer(&_timer, _timerQueue, (WAITORTIMERCALLBACK)RTthread, reinterpret_cast<void*>(this), 0, 1, 0))
				std::cerr << "CreateTimerQueueTimer failed." << std::endl;

			ec_writestate(0);

			auto chk = 200;
			do
			{
				ec_statecheck(0, EC_STATE_OPERATIONAL, 50000);
			} while (chk-- && (ec_slave[0].state != EC_STATE_OPERATIONAL));

			if (ec_slave[0].state == EC_STATE_OPERATIONAL)
			{
				_isClosed = false;
			}
			else {
				if (!DeleteTimerQueueTimer(_timerQueue, _timer, 0))
					std::cerr << "DeleteTimerQueue failed." << std::endl;
				std::cerr << "One ore more slaves are not responding." << std::endl;
			}
		}
		else
		{
			std::cerr << "No slaves found!" << std::endl;
		}
	}
	else
	{
		std::cerr << "No socket connection on " << ifname << std::endl;
	}
}

void SOEMController::impl::Close()
{
	if (!_isClosed) {
		if (!DeleteTimerQueueTimer(_timerQueue, _timer, 0))
			std::cerr << "DeleteTimerQueue failed." << std::endl;

		ec_slave[0].state = EC_STATE_INIT;
		ec_writestate(0);

		ec_close();

		_isClosed = true;
	}
}

SOEMController::SOEMController()
{
	this->_pimpl = std::make_shared<impl>();
}

SOEMController::~SOEMController()
{
	this->_pimpl->Close();
}

void SOEMController::Open(const char* ifname, size_t devNum)
{
	this->_pimpl->Open(ifname, devNum);
}

void SOEMController::Send(size_t size, std::unique_ptr<uint8_t[]> buf)
{
	this->_pimpl->Send(size, std::move(buf));
}

void SOEMController::Close()
{
	this->_pimpl->Close();
}