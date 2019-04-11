/*
*  autd3.cpp
*  autd3
*
*  Created by Seki Inoue on 5/13/16.
*  Modified by Shun Suzuki on 02/07/2018.
*  Modified by Shun Suzuki on 04/11/2019.
*  Copyright © 2018 Hapis Lab. All rights reserved.
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
#include <codeanalysis\warnings.h>
#pragma warning( push )
#pragma warning ( disable : ALL_CODE_ANALYSIS_WARNINGS )
#include <boost/algorithm/string.hpp>
#include <boost/assert.hpp>
#pragma warning( pop )
#include "link.hpp"
#include "controller.hpp"
#include "privdef.hpp"
#include "ethercat_link.hpp"
#include "lateraltimer.hpp"

class autd::Controller::impl {
	friend class autd::Controller::lateraltimer;

public:
	GeometryPtr _geometry;
	std::shared_ptr<internal::Link> _link;
	std::queue<GainPtr> _build_q;
	std::queue<GainPtr> _send_gain_q;
	std::queue<ModulationPtr> _send_mod_q;

	std::thread _build_thr;
	std::thread _send_thr;
	std::condition_variable _build_cond;
	std::condition_variable _send_cond;
	std::mutex _build_mtx;
	std::mutex _send_mtx;

	int8_t frequency_shift;
	bool silentMode;
	uint8_t modReset;
	bool LMrunning;

	~impl();
	bool isOpen();
	void Close();

	void InitPipeline();
	void AppendGain(const GainPtr gain);
	void AppendGainSync(const GainPtr gain);
	void AppendModulation(const ModulationPtr mod);
	void AppendModulationSync(const ModulationPtr mod);

	void FlushBuffer();
	std::unique_ptr<uint8_t[]> MakeBody(GainPtr gain, ModulationPtr mod, size_t *size);
};

class autd::Controller::lateraltimer : public Timer {
	friend class autd::Controller;

public:
	lateraltimer();
	void AppendLateralGain(autd::GainPtr gain, const GeometryPtr geometry);
	void AppendLateralGain(const std::vector<GainPtr> &gain_list, const GeometryPtr geometry);
	void StartLateralModulation(float freq);
	void FinishLateralModulation();
	void ResetLateralGain();
	int Size();
protected:
	void Run();
private:
	std::shared_ptr<autd::Controller::impl> _pcnt;
	int _lateral_gain_size;
	int _lateral_gain_idx;
	std::vector<autd::GainPtr> _lateral_gain;
	bool _runnig;
};

autd::Controller::lateraltimer::lateraltimer() {
	this->_lateral_gain_size = 0;
	this->_lateral_gain_idx = 0;
}

int autd::Controller::lateraltimer::Size() {
	return this->_lateral_gain_size;
}

void autd::Controller::lateraltimer::Run() {
	try {
		size_t body_size = 0;
		auto gain = this->_lateral_gain[this->_lateral_gain_idx];
		this->_lateral_gain_idx = (this->_lateral_gain_idx + 1) % this->_lateral_gain_size;
		this->_pcnt->AppendGainSync(gain);
	}
	catch (int errnum) {
		this->_pcnt->Close();
		std::cerr << errnum << "Link closed." << std::endl;
	}
}

void autd::Controller::lateraltimer::StartLateralModulation(float freq)
{
	if (this->Size() == 0) {
		std::cerr << "Call \"AppendLateralGain\" before start Lateral Modulation" << std::endl;
		return;
	}

	this->FinishLateralModulation();

	const int itvl_micro_sec = static_cast<int>(1000 * 1000 / freq / this->Size());
	this->SetInterval(itvl_micro_sec);
	this->Start();
	this->_runnig = true;
}

void autd::Controller::lateraltimer::AppendLateralGain(autd::GainPtr gain, const GeometryPtr geometry) {
	gain->SetGeometry(geometry);
	if (!gain->built()) gain->build();

	this->_lateral_gain_size++;
	this->_lateral_gain_idx = 0;
	this->_lateral_gain.push_back(gain);
}

void autd::Controller::lateraltimer::AppendLateralGain(const std::vector<GainPtr> &gain_list, const GeometryPtr geometry)
{
	for (GainPtr g : gain_list) {
		this->AppendLateralGain(g, geometry);
	}
}

void autd::Controller::lateraltimer::FinishLateralModulation() {
	if (this->_runnig) this->Stop();
	this->_runnig = false;
}

void autd::Controller::lateraltimer::ResetLateralGain()
{
	this->FinishLateralModulation();
	std::vector<autd::GainPtr>().swap(this->_lateral_gain);
}

autd::Controller::impl::~impl() {
	if (std::this_thread::get_id() != this->_build_thr.get_id() && this->_build_thr.joinable()) this->_build_thr.join();
	if (std::this_thread::get_id() != this->_send_thr.get_id() && this->_send_thr.joinable()) this->_send_thr.join();
}

void autd::Controller::impl::InitPipeline() {
	// pipeline step #1
	this->_build_thr = std::thread([&] {
		while (this->isOpen()) {
			GainPtr gain = nullptr;
			// wait for gain
			{
				std::unique_lock<std::mutex> lk(_build_mtx);
				_build_cond.wait(lk, [&] {
					return _build_q.size() || !this->isOpen();
					});
				if (_build_q.size()) {
					gain = _build_q.front();
					_build_q.pop();
				}
			}

			// build gain
			if (gain != nullptr && !gain->built()) gain->build();
			if (gain != nullptr && gain->_fix) gain->FixImpl();

			// pass gain to next pipeline stage
			{
				std::unique_lock<std::mutex> lk(_send_mtx);
				_send_gain_q.push(gain);
				_send_cond.notify_all();
			}
		}
		});

	// pipeline step #2
	this->_send_thr = std::thread([&] {
		try {
			while (this->isOpen()) {
				GainPtr gain = nullptr;
				ModulationPtr mod = nullptr;

				// wait for inputs
				{
					std::unique_lock<std::mutex> lk(_send_mtx);
					_send_cond.wait(lk, [&] {
						return _send_gain_q.size() || _send_mod_q.size() || !this->isOpen();
						});
					if (_send_gain_q.size())
						gain = _send_gain_q.front();
					if (_send_mod_q.size())
						mod = _send_mod_q.front();
				}
#ifdef DEBUG
				auto start = std::chrono::steady_clock::now();
#endif

				size_t body_size = 0;
				auto body = MakeBody(gain, mod, &body_size);
				if (this->_link->isOpen()) this->_link->Send(body_size, std::move(body));
#ifdef DEBUG
				auto end = std::chrono::steady_clock::now();
				std::cout << std::chrono::duration <double, std::milli>(end - start).count() << " ms" << std::endl;
#endif
				// remove elements
				std::unique_lock<std::mutex> lk(_send_mtx);
				if (gain != nullptr)
					_send_gain_q.pop();
				if (mod != nullptr && mod->buffer.size() <= mod->sent) {
					mod->sent = 0;
					_send_mod_q.pop();
				}

				std::this_thread::sleep_for(std::chrono::milliseconds(1));
			}
		}
		catch (const int errnum) {
			this->Close();
			std::cerr << errnum << "Link closed." << std::endl;
		}
		});
}

void autd::Controller::impl::AppendGain(GainPtr gain) {
	{
		gain->SetGeometry(this->_geometry);
		std::unique_lock<std::mutex> lk(_build_mtx);
		_build_q.push(gain);
	}
	_build_cond.notify_all();
}

void autd::Controller::impl::AppendGainSync(autd::GainPtr gain) {
	try {
		gain->SetGeometry(this->_geometry);
		if (!gain->built()) gain->build();
		if (gain->_fix) gain->FixImpl();

		size_t body_size = 0;

		auto body = this->MakeBody(gain, nullptr, &body_size);
		if (this->isOpen()) this->_link->Send(body_size, std::move(body));
	}
	catch (const int errnum) {
		this->_link->Close();
		std::cerr << errnum << "Link closed." << std::endl;
	}
}

void autd::Controller::impl::AppendModulation(autd::ModulationPtr mod) {
	std::unique_lock<std::mutex> lk(_send_mtx);
	_send_mod_q.push(mod);
	_send_cond.notify_all();
}

void autd::Controller::impl::AppendModulationSync(autd::ModulationPtr mod) {
	try {
		if (this->isOpen()) {
			while (mod->buffer.size() > mod->sent) {
				size_t body_size = 0;
				auto body = this->MakeBody(nullptr, mod, &body_size);
				this->_link->Send(body_size, std::move(body));
				std::this_thread::sleep_for(std::chrono::milliseconds(1));
			}
			mod->sent = 0;
		}
	}
	catch (const int errnum) {
		this->Close();
		std::cerr << errnum << "Link closed." << std::endl;
	}
}

void autd::Controller::impl::FlushBuffer() {
	std::unique_lock<std::mutex> lk0(_send_mtx);
	std::unique_lock<std::mutex> lk1(_build_mtx);
	std::queue<GainPtr>().swap(_build_q);
	std::queue<GainPtr>().swap(_send_gain_q);
	std::queue<ModulationPtr>().swap(_send_mod_q);
}

std::unique_ptr<uint8_t[]> autd::Controller::impl::MakeBody(GainPtr gain, ModulationPtr mod, size_t *size) {
	auto num_devices = (gain != nullptr) ? gain->geometry()->numDevices() : 0;

	*size = sizeof(RxGlobalHeader) + sizeof(uint16_t)*NUM_TRANS_IN_UNIT*num_devices;
	auto body = std::make_unique<uint8_t[]>(*size);

	auto *header = reinterpret_cast<RxGlobalHeader*>(&body[0]);
	header->msg_id = static_cast<uint8_t>(rand() % 256); // NOLINT
	header->control_flags = 0;
	header->mod_size = 0;

	if (this->silentMode) header->control_flags |= SILENT;

	if (mod != nullptr) {
		header->control_flags |= (this->modReset ^= MOD_RESET);

		const auto mod_size = std::max(0, std::min(static_cast<int>(mod->buffer.size() - mod->sent), MOD_FRAME_SIZE));
		header->mod_size = mod_size;
		if (mod->sent == 0) header->control_flags |= MOD_BEGIN;
		if (mod->loop && mod->sent == 0) header->control_flags |= LOOP_BEGIN;
		if (mod->loop && mod->sent + mod_size >= mod->buffer.size()) header->control_flags |= LOOP_END;
		header->frequency_shift = this->_geometry->_freq_shift;

		memcpy(header->mod, &mod->buffer[mod->sent], mod_size);
		mod->sent += mod_size;
	}

	auto *cursor = &body[0] + sizeof(RxGlobalHeader) / sizeof(body[0]);
	if (gain != nullptr) {
		for (int i = 0; i < gain->geometry()->numDevices(); i++) {
			auto deviceId = gain->geometry()->deviceIdForDeviceIdx(i);
			auto byteSize = NUM_TRANS_IN_UNIT * sizeof(uint16_t);
			memcpy(cursor, &gain->_data[deviceId][0], byteSize);
			cursor += byteSize / sizeof(body[0]);
		}
	}
	return body;
}


bool autd::Controller::impl::isOpen() {
	return this->_link.get() && this->_link->isOpen();
}

void autd::Controller::impl::Close() {
	if (this->isOpen()) {
		auto nullgain = NullGain::Create();
		this->AppendGainSync(nullgain);
#if DLL_FOR_CSHARP
		delete nullgain;
#endif
		this->_link->Close();
		this->FlushBuffer();
		this->_build_cond.notify_all();
		if (std::this_thread::get_id() != this->_build_thr.get_id() && this->_build_thr.joinable()) this->_build_thr.join();
		this->_send_cond.notify_all();
		if (std::this_thread::get_id() != this->_send_thr.get_id() && this->_send_thr.joinable()) this->_send_thr.join();
		this->_link = std::shared_ptr<internal::EthercatLink>(nullptr);
	}
}

autd::Controller::Controller() {
	this->_pimpl = std::make_shared<impl>();
	this->_pimpl->_geometry = std::make_shared<Geometry>();
	this->_pimpl->frequency_shift = -3;
	this->_pimpl->silentMode = true;
	this->_pimpl->modReset = MOD_RESET;
	this->_pimpl->LMrunning = false;

	this->_ptimer = std::make_unique<lateraltimer>();
	this->_ptimer->_pcnt = this->_pimpl;
}

autd::Controller::~Controller() {
	this->Close();
}

void autd::Controller::Open(autd::LinkType type, std::string location) {
	this->Close();

	switch (type) {
	case LinkType::ETHERCAT: {
		// TODO(volunteer): a smarter localhost detection
		if (location == "" ||
			location.find("localhost") == 0 ||
			location.find("0.0.0.0") == 0 ||
			location.find("127.0.0.1") == 0) {
			this->_pimpl->_link = std::make_shared<internal::LocalEthercatLink>();
		}
		else {
			this->_pimpl->_link = std::make_shared<internal::EthercatLink>();
		}
		this->_pimpl->_link->Open(location);
		break;
	}
	default:
		BOOST_ASSERT_MSG(false, "This link type is not implemented yet.");
		break;
	}

	if (this->_pimpl->_link->isOpen())
		this->_pimpl->InitPipeline();
	else
		this->Close();
}

bool autd::Controller::isOpen() {
	return this->_pimpl->isOpen();
}

void autd::Controller::Close() {
	this->_pimpl->Close();
}

void autd::Controller::AppendGain(GainPtr gain) {
	this->_ptimer->FinishLateralModulation();
	this->_pimpl->AppendGain(gain);
}

void autd::Controller::AppendGainSync(GainPtr gain) {
	this->_ptimer->FinishLateralModulation();
	this->_pimpl->AppendGainSync(gain);
}

void autd::Controller::AppendModulation(ModulationPtr modulation) {
	this->_pimpl->AppendModulation(modulation);
}

void autd::Controller::AppendModulationSync(ModulationPtr modulation) {
	this->_pimpl->AppendModulationSync(modulation);
}

void autd::Controller::AppendLateralGain(GainPtr gain)
{
	this->_ptimer->AppendLateralGain(gain, this->geometry());
}

void autd::Controller::AppendLateralGain(const std::vector<GainPtr> &gain_list)
{
	this->_ptimer->AppendLateralGain(gain_list, this->geometry());
}

void autd::Controller::StartLateralModulation(float freq)
{
	this->_ptimer->StartLateralModulation(freq);
}

void autd::Controller::FinishLateralModulation()
{
	this->_ptimer->FinishLateralModulation();
}

void autd::Controller::ResetLateralGain()
{
	this->_ptimer->ResetLateralGain();
}

void autd::Controller::Flush() {
	this->_pimpl->FlushBuffer();
}

autd::Controller& autd::Controller::operator<<(const uint8_t coef) {
	auto mod = Modulation::Create();
	mod->buffer.push_back(coef);
	mod->loop = true;
	this->AppendModulation(mod);
	return *this;
}

autd::Controller& autd::Controller::operator<<(ModulationPtr mod) {
	this->AppendModulation(mod);
	return *this;
}

autd::Controller& autd::Controller::operator<<(GainPtr gain) {
	this->AppendGain(gain);
	return *this;
}

autd::GeometryPtr autd::Controller::geometry() noexcept {
	return this->_pimpl->_geometry;
}

void autd::Controller::SetGeometry(const GeometryPtr &geometry) noexcept {
	this->_pimpl->_geometry = geometry;
}

size_t autd::Controller::remainingInBuffer() {
	return this->_pimpl->_send_gain_q.size() + this->_pimpl->_send_mod_q.size() + this->_pimpl->_build_q.size();
}

void autd::Controller::SetSilentMode(bool silent) noexcept {
	this->_pimpl->silentMode = silent;
}

bool autd::Controller::silentMode() noexcept {
	return this->_pimpl->silentMode;
}
