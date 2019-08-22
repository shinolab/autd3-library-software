#include <memory>
#include <iostream>

#include "libsoem.hpp"
#include "ethercat.h"

using namespace libsoem;

class SOEMController::impl
{
public:
	void Open(const char* ifname, size_t devNum);
	void Send(size_t size, std::unique_ptr<uint8_t[]> buf);
	void Close();

private:
	uint8_t* _IOmap;
	bool _isClosed;
	size_t _devNum;
};

void SOEMController::impl::Send(size_t size, std::unique_ptr<uint8_t[]> buf)
{
	const auto header_size = MOD_SIZE + 4;
	const auto data_size = TRANS_NUM * 2;
	const auto includes_gain = ((size - header_size) / data_size) > 0;

	//ec_configdc();

	for (size_t i = 0; i < _devNum; i++)
	{
		if (includes_gain) memcpy(&_IOmap[FRAME_SIZE * i], &buf[header_size + data_size * i], TRANS_NUM * 2);
		memcpy(&_IOmap[FRAME_SIZE * i + TRANS_NUM * 2], &buf[0], MOD_SIZE + 4);
	}

	/*std::cout << "**************************************" << std::endl;
	for (size_t i = 0; i < FRAME_SIZE * _devNum; i++)
	{
		std::cout << i << "," << (int)_IOmap[i];
	}*/

	ec_send_processdata();
	//ec_receive_processdata(EC_TIMEOUTRET);

	//ec_config_map(_IOmap);
}

void SOEMController::impl::Open(const char* ifname, size_t devNum)
{
	_devNum = devNum;
	auto size = FRAME_SIZE * _devNum;
	_IOmap = new uint8_t[size];

	if (ec_init(ifname))
	{
		if (ec_config_init(0) > 0)
		{
			ec_config_map(_IOmap);
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
				_isClosed = false;
			}
			else {

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
		ec_slave[0].state = EC_STATE_INIT;
		ec_writestate(0);

		ec_close();

		delete[] _IOmap;

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