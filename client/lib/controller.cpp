/*
 * File: controller.cpp
 * Project: lib
 * Created Date: 13/05/2016
 * Author: Seki Inoue
 * -----
 * Last Modified: 17/02/2020
 * Modified By: Shun Suzuki (suzuki@hapis.k.u-tokyo.ac.jp)
 * -----
 * Copyright (c) 2016-2019 Hapis Lab. All rights reserved.
 *
 */

#include <iostream>
#include <string>
#include <algorithm>
#include <vector>
#include <queue>
#include <condition_variable>
#include <mutex>
#include <thread>
#include <chrono>

#include "link.hpp"
#include "controller.hpp"
#include "geometry.hpp"
#include "privdef.hpp"
#if WIN32
#include "ethercat_link.hpp"
#endif
#include "soem_link.hpp"
#include "timer.hpp"

using namespace autd;
using namespace std;

#pragma region Controller::impl
class Controller::impl
{
public:
	GeometryPtr _geometry;
	shared_ptr<internal::Link> _link;
	queue<GainPtr> _build_q;
	queue<GainPtr> _send_gain_q;
	queue<ModulationPtr> _send_mod_q;
	queue<GainPtr> _stmGains;
	vector<uint8_t *> _stmBodies;
	unique_ptr<Timer> _pStmTimer;

	thread _build_thr;
	thread _send_thr;
	condition_variable _build_cond;
	condition_variable _send_cond;
	mutex _build_mtx;
	mutex _send_mtx;

	bool silentMode = true;
	bool isOpen();

	impl();
	~impl();
	void CalibrateModulation();
	void Close();

	void InitPipeline();
	void Stop();
	void AppendGain(const GainPtr gain);
	void AppendGainSync(const GainPtr gain);
	void AppendModulation(const ModulationPtr mod);
	void AppendModulationSync(const ModulationPtr mod);
	void AppendSTMGain(GainPtr gain);
	void AppendSTMGain(const std::vector<GainPtr> &gain_list);
	void StartSTModulation(float freq);
	void StopSTModulation();
	void FinishSTModulation();
	void FlushBuffer();

	unique_ptr<uint8_t[]> MakeBody(GainPtr gain, ModulationPtr mod, size_t *size);

	static uint8_t get_id()
	{
		static atomic<uint8_t> id{0};

		id.fetch_add(0x01);
		uint8_t expected = 0xf0;
		id.compare_exchange_weak(expected, 1);

		return id.load();
	}
};

Controller::impl::impl()
{
	this->_geometry = Geometry::Create();
	this->silentMode = true;
	this->_pStmTimer = std::make_unique<Timer>();
}

Controller::impl::~impl()
{
	if (this_thread::get_id() != this->_build_thr.get_id() && this->_build_thr.joinable())
		this->_build_thr.join();
	if (this_thread::get_id() != this->_send_thr.get_id() && this->_send_thr.joinable())
		this->_send_thr.join();
}

void Controller::impl::InitPipeline()
{
	this->_build_thr = thread([&] {
		while (this->isOpen())
		{
			GainPtr gain = nullptr;
			{
				unique_lock<mutex> lk(_build_mtx);

				_build_cond.wait(lk, [&] {
					return _build_q.size() || !this->isOpen();
				});

				if (_build_q.size() > 0)
				{
					gain = _build_q.front();
					_build_q.pop();
				}
			}

			if (gain != nullptr)
			{
				if (!gain->built())
					gain->build();
				{
					unique_lock<mutex> lk(_send_mtx);
					_send_gain_q.push(gain);
					_send_cond.notify_all();
				}
			}
		}
	});

	this->_send_thr = thread([&] {
		try
		{
			while (this->isOpen())
			{
				GainPtr gain = nullptr;
				ModulationPtr mod = nullptr;

				{
					unique_lock<mutex> lk(_send_mtx);
					_send_cond.wait(lk, [&] {
						return _send_gain_q.size() || _send_mod_q.size() || !this->isOpen();
					});
					if (_send_gain_q.size() > 0)
						gain = _send_gain_q.front();
					if (_send_mod_q.size() > 0)
						mod = _send_mod_q.front();
				}
				size_t body_size = 0;
				auto body = MakeBody(gain, mod, &body_size);
				if (this->_link->isOpen())
					this->_link->Send(body_size, move(body));

				unique_lock<mutex> lk(_send_mtx);
				if (gain != nullptr && _send_gain_q.size() > 0)
					_send_gain_q.pop();
				if (mod != nullptr && mod->buffer.size() <= mod->sent)
				{
					mod->sent = 0;
					if (_send_mod_q.size() > 0)
						_send_mod_q.pop();
				}
			}
		}
		catch (const int errnum)
		{
			this->Close();
			cerr << errnum << "Link closed." << endl;
		}
	});
}

void Controller::impl::Stop()
{
	auto nullgain = NullGain::Create();
	this->AppendGainSync(nullgain);
#if DLL_FOR_CAPI
	delete nullgain;
#endif
}

void Controller::impl::AppendGain(GainPtr gain)
{
	{
		gain->SetGeometry(this->_geometry);
		unique_lock<mutex> lk(_build_mtx);
		_build_q.push(gain);
	}
	_build_cond.notify_all();
}

void Controller::impl::AppendGainSync(GainPtr gain)
{
	try
	{
		gain->SetGeometry(this->_geometry);
		if (!gain->built())
			gain->build();

		size_t body_size = 0;
		auto body = this->MakeBody(gain, nullptr, &body_size);

		if (this->isOpen())
			this->_link->Send(body_size, move(body));
	}
	catch (const int errnum)
	{
		this->_link->Close();
		cerr << errnum << "Link closed." << endl;
	}
}

void Controller::impl::AppendModulation(ModulationPtr mod)
{
	unique_lock<mutex> lk(_send_mtx);
	_send_mod_q.push(mod);
	_send_cond.notify_all();
}

void Controller::impl::AppendModulationSync(ModulationPtr mod)
{
	try
	{
		if (this->isOpen())
		{
			while (mod->buffer.size() > mod->sent)
			{
				size_t body_size = 0;
				auto body = this->MakeBody(nullptr, mod, &body_size);

				this->_link->Send(body_size, move(body));
				this_thread::sleep_for(chrono::milliseconds(1));
			}
			mod->sent = 0;
		}
	}
	catch (const int errnum)
	{
		this->Close();
		cerr << errnum << "Link closed." << endl;
	}
}

void Controller::impl::AppendSTMGain(GainPtr gain)
{
	_stmGains.push(gain);
}

void Controller::impl::AppendSTMGain(const std::vector<GainPtr> &gain_list)
{
	for (auto g : gain_list)
	{
		this->AppendSTMGain(g);
	}
}

void Controller::impl::StartSTModulation(float freq)
{
	auto len = this->_stmGains.size();
	auto itvl_us = static_cast<int>(1000000. / freq / len);
	this->_pStmTimer->SetInterval(itvl_us);

	vector<size_t> bodysizes;
	this->_stmBodies.reserve(len);
	bodysizes.reserve(len);

	for (size_t i = 0; i < len; i++)
	{
		auto g = this->_stmGains.front();
		g->SetGeometry(this->_geometry);
		if (!g->built())
			g->build();

		size_t body_size = 0;
		auto body = this->MakeBody(g, nullptr, &body_size);
		uint8_t *b = new uint8_t[body_size];
		std::memcpy(b, body.get(), body_size);
		this->_stmBodies.push_back(b);
		bodysizes.push_back(body_size);

		this->_stmGains.pop();
	}

	size_t idx = 0;
	this->_pStmTimer->Start(
		[this, idx, len, bodysizes]() mutable {
			auto body_size = bodysizes[idx];
			auto body_copy = make_unique<uint8_t[]>(body_size);
			uint8_t *p = this->_stmBodies[idx];
			std::memcpy(body_copy.get(), p, body_size);
			if (this->isOpen())
				this->_link->Send(body_size, move(body_copy));
			idx = (idx + 1) % len;
		});
}

void Controller::impl::StopSTModulation()
{
	this->_pStmTimer->Stop();
}

void Controller::impl::FinishSTModulation()
{
	this->StopSTModulation();
	queue<GainPtr>().swap(this->_stmGains);
	for (uint8_t *p : this->_stmBodies)
	{
		delete[] p;
	}
	vector<uint8_t *>().swap(this->_stmBodies);
}

void Controller::impl::CalibrateModulation()
{
	this->_link->CalibrateModulation();
}

void Controller::impl::FlushBuffer()
{
	unique_lock<mutex> lk0(_send_mtx);
	unique_lock<mutex> lk1(_build_mtx);
	queue<GainPtr>().swap(_build_q);
	queue<GainPtr>().swap(_send_gain_q);
	queue<ModulationPtr>().swap(_send_mod_q);
}

unique_ptr<uint8_t[]> Controller::impl::MakeBody(GainPtr gain, ModulationPtr mod, size_t *size)
{
	auto num_devices = (gain != nullptr) ? gain->geometry()->numDevices() : 0;

	*size = sizeof(RxGlobalHeader) + sizeof(uint16_t) * NUM_TRANS_IN_UNIT * num_devices;
	auto body = make_unique<uint8_t[]>(*size);

	auto *header = reinterpret_cast<RxGlobalHeader *>(&body[0]);
	header->msg_id = get_id();
	header->control_flags = 0;
	header->mod_size = 0;

	if (this->silentMode)
		header->control_flags |= SILENT;

	if (mod != nullptr)
	{
		const uint8_t mod_size = max(0, min(static_cast<int>(mod->buffer.size() - mod->sent), MOD_FRAME_SIZE));
		header->mod_size = mod_size;
		if (mod->sent == 0)
			header->control_flags |= LOOP_BEGIN;
		if (mod->sent + mod_size >= mod->buffer.size())
			header->control_flags |= LOOP_END;
		header->frequency_shift = this->_geometry->_freq_shift;

		std::memcpy(header->mod, &mod->buffer[mod->sent], mod_size);
		mod->sent += mod_size;
	}

	auto *cursor = &body[0] + sizeof(RxGlobalHeader) / sizeof(body[0]);
	if (gain != nullptr)
	{
		for (int i = 0; i < gain->geometry()->numDevices(); i++)
		{
			auto deviceId = gain->geometry()->deviceIdForDeviceIdx(i);
			auto byteSize = NUM_TRANS_IN_UNIT * sizeof(uint16_t);
			std::memcpy(cursor, &gain->_data[deviceId].at(0), byteSize);
			cursor += byteSize / sizeof(body[0]);
		}
	}
	return body;
}

bool Controller::impl::isOpen()
{
	return this->_link.get() && this->_link->isOpen();
}

void Controller::impl::Close()
{
	if (this->isOpen())
	{
		this->FinishSTModulation();
		this->Stop();
		this->_link->Close();
		this->FlushBuffer();
		this->_build_cond.notify_all();
		if (this_thread::get_id() != this->_build_thr.get_id() && this->_build_thr.joinable())
			this->_build_thr.join();
		this->_send_cond.notify_all();
		if (this_thread::get_id() != this->_send_thr.get_id() && this->_send_thr.joinable())
			this->_send_thr.join();
		this->_link = shared_ptr<internal::Link>(nullptr);
	}
}
#pragma endregion

#pragma region pimpl
Controller::Controller()
{
	this->_pimpl = std::make_unique<impl>();
}

Controller::~Controller() noexcept(false)
{
	this->Close();
}

bool Controller::isOpen()
{
	return this->_pimpl->isOpen();
}

GeometryPtr Controller::geometry() noexcept
{
	return this->_pimpl->_geometry;
}

bool Controller::silentMode() noexcept
{
	return this->_pimpl->silentMode;
}

size_t Controller::remainingInBuffer()
{
	return this->_pimpl->_send_gain_q.size() + this->_pimpl->_send_mod_q.size() + this->_pimpl->_build_q.size();
}

EtherCATAdapters Controller::EnumerateAdapters(int &size)
{
	auto adapters = libsoem::EtherCATAdapterInfo::EnumerateAdapters();
	size = static_cast<int>(adapters.size());
#if DLL_FOR_CAPI
	EtherCATAdapters res = new EtherCATAdapter[size];
	int i = 0;
#else
	EtherCATAdapters res;
#endif
	for (auto adapter : libsoem::EtherCATAdapterInfo::EnumerateAdapters())
	{
		EtherCATAdapter p;
#if DLL_FOR_CAPI
		p.first = *adapter.desc.get();
		p.second = *adapter.name.get();
		res[i++] = p;
#else
		p.first = adapter.desc;
		p.second = adapter.name;
		res.push_back(p);
#endif
	}
	return res;
}

void Controller::Open(LinkType type, string location)
{
	this->Close();

	switch (type)
	{
#if WIN32
	case LinkType::ETHERCAT:
	case LinkType::TwinCAT:
	{
		// TODO(volunteer): a smarter localhost detection
		if (location == "" ||
			location.find("localhost") == 0 ||
			location.find("0.0.0.0") == 0 ||
			location.find("127.0.0.1") == 0)
		{
			this->_pimpl->_link = make_shared<internal::LocalEthercatLink>();
		}
		else
		{
			this->_pimpl->_link = make_shared<internal::EthercatLink>();
		}
		this->_pimpl->_link->Open(location);
		break;
	}
#endif
	case LinkType::SOEM:
	{
		this->_pimpl->_link = make_shared<internal::SOEMLink>();
		auto devnum = this->_pimpl->_geometry->numDevices();
		this->_pimpl->_link->Open(location + ":" + to_string(devnum));
		break;
	}
	default:
		throw runtime_error("This link type is not implemented yet.");
		break;
	}

	if (this->_pimpl->_link->isOpen())
		this->_pimpl->InitPipeline();
	else
		this->Close();
}

void Controller::SetSilentMode(bool silent) noexcept
{
	this->_pimpl->silentMode = silent;
}

void Controller::CalibrateModulation()
{
	this->_pimpl->CalibrateModulation();
}

void Controller::Close()
{
	this->_pimpl->Close();
}

void Controller::Stop()
{
	this->_pimpl->Stop();
}

void Controller::AppendGain(GainPtr gain)
{
	this->_pimpl->AppendGain(gain);
}

void Controller::AppendGainSync(GainPtr gain)
{
	this->_pimpl->AppendGainSync(gain);
}

void Controller::AppendModulation(ModulationPtr modulation)
{
	this->_pimpl->AppendModulation(modulation);
}

void Controller::AppendModulationSync(ModulationPtr modulation)
{
	this->_pimpl->AppendModulationSync(modulation);
}

void Controller::AppendSTMGain(GainPtr gain)
{
	this->_pimpl->AppendSTMGain(gain);
}

void Controller::AppendSTMGain(const vector<GainPtr> &gain_list)
{
	this->_pimpl->AppendSTMGain(gain_list);
}

void Controller::StartSTModulation(float freq)
{
	this->_pimpl->StartSTModulation(freq);
}

void Controller::StopSTModulation()
{
	this->_pimpl->StopSTModulation();
}

void Controller::FinishSTModulation()
{
	this->_pimpl->FinishSTModulation();
}

void Controller::Flush()
{
	this->_pimpl->FlushBuffer();
}

void Controller::LateralModulationAT(Eigen::Vector3f point, Eigen::Vector3f dir, float lm_amp, float lm_freq)
{
	auto p1 = point + lm_amp * dir;
	auto p2 = point - lm_amp * dir;
	this->FinishSTModulation();
	this->AppendSTMGain(autd::FocalPointGain::Create(p1));
	this->AppendSTMGain(autd::FocalPointGain::Create(p2));
	this->StartSTModulation(lm_freq);
}

#pragma region deprecated
void Controller::AppendLateralGain(GainPtr gain)
{
	this->AppendSTMGain(gain);
}
void Controller::AppendLateralGain(const std::vector<GainPtr> &gain_list)
{
	this->AppendSTMGain(gain_list);
}
void Controller::StartLateralModulation(float freq)
{
	this->StartSTModulation(freq);
}
void Controller::FinishLateralModulation()
{
	this->StopSTModulation();
}
void Controller::ResetLateralGain()
{
	this->FinishSTModulation();
}
#pragma endregion

#pragma endregion
