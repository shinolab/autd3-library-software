/*
*  autd3.cpp
*  autd3
*
*  Created by Seki Inoue on 5/13/16.
*  Changed by Shun Suzuki on 02/07/2018.
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
#include "autd3.hpp"
#include "privdef.hpp"
#include "ethercat_link.hpp"
#include "timer.hpp"

class autd::Controller::impl {
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

	void AppendLateralGain(GainPtr gain);
	void AppendLateralGain(const std::vector<GainPtr> &gain_list);
	void StartLateralModulation(float freq);
	void FinishLateralModulation();
	void ResetLateralGain();

	void FlushBuffer();
	std::unique_ptr<uint8_t[]> MakeBody(GainPtr gain, ModulationPtr mod, size_t *size);

	class LateralTimer;
	std::unique_ptr<LateralTimer> _plt;
};

class autd::Controller::impl::LateralTimer : public Timer {
public:
	LateralTimer(std::shared_ptr<internal::Link> link, GeometryPtr geometry, bool silentMode);
	void AppendLateralGain(autd::GainPtr gain);
	int Size();
protected:
	void Run();
private:
	int _lateral_gain_size;
	int _lateral_gain_idx;
	std::vector<autd::GainPtr> _lateral_gain;
	std::shared_ptr<internal::Link> _link;
	GeometryPtr _geometry;
	bool _silentMode;

	bool isOpen();
	void MakeAndSendLateralGain();
	std::unique_ptr<uint8_t[]> MakeBody(GainPtr gain, ModulationPtr mod, size_t *size);
};

autd::Controller::impl::LateralTimer::LateralTimer(std::shared_ptr<internal::Link> link, GeometryPtr geometry, bool silentMode) {
	this->_lateral_gain_size = 0;
	this->_lateral_gain_idx = 0;
	this->_link = link;
	this->_geometry = geometry;
	this->_silentMode = silentMode;
}

void autd::Controller::impl::LateralTimer::AppendLateralGain(autd::GainPtr gain) {
	this->_lateral_gain_size++;
	this->_lateral_gain_idx = 0;
	this->_lateral_gain.push_back(gain);
}

int autd::Controller::impl::LateralTimer::Size() {
	return this->_lateral_gain_size;
}

void autd::Controller::impl::LateralTimer::Run() {
	MakeAndSendLateralGain();
}

bool autd::Controller::impl::LateralTimer::isOpen() {
	return this->_link.get() && this->_link->isOpen();
}

void autd::Controller::impl::LateralTimer::MakeAndSendLateralGain() {
	try {
		size_t body_size = 0;
		auto gain = this->_lateral_gain[this->_lateral_gain_idx];
		this->_lateral_gain_idx = (this->_lateral_gain_idx + 1) % this->_lateral_gain_size;
		std::unique_ptr<uint8_t[]> body = this->MakeBody(gain, nullptr, &body_size);
		if (this->isOpen()) this->_link->Send(body_size, std::move(body));
	}
	catch (int errnum) {
		this->_link->Close();
		std::cerr << errnum << "Link closed." << std::endl;
	}
}

std::unique_ptr<uint8_t[]> autd::Controller::impl::LateralTimer::MakeBody(GainPtr gain, ModulationPtr mod, size_t *size) {
	int num_devices = (gain != nullptr) ? gain->geometry()->numDevices() : 0;

	*size = sizeof(RxGlobalHeader) + sizeof(uint16_t)*NUM_TRANS_IN_UNIT*num_devices;
	std::unique_ptr<uint8_t[]> body(new uint8_t[*size]);

	RxGlobalHeader *header = reinterpret_cast<RxGlobalHeader*>(&body[0]);
	header->msg_id = static_cast<uint8_t>(rand() % 256); // NOLINT
	header->control_flags = 0;
	header->mod_size = 0;

	if (this->_silentMode)header->control_flags |= SILENT;

	if (mod != nullptr) {
		int mod_size = std::max(0, std::min(static_cast<int>(mod->buffer.size() - mod->sent), MOD_FRAME_SIZE));
		header->mod_size = mod_size;
		if (mod->sent == 0) header->control_flags |= MOD_BEGIN;
		if (mod->loop && mod->sent == 0) header->control_flags |= LOOP_BEGIN;
		if (mod->loop && mod->sent + mod_size >= mod->buffer.size()) header->control_flags |= LOOP_END;
		header->frequency_shift = this->_geometry->_freq_shift;

		memcpy(header->mod, &mod->buffer[mod->sent], mod_size);
		mod->sent += mod_size;
	}

	uint8_t *cursor = &body[0] + sizeof(RxGlobalHeader) / sizeof(body[0]);
	if (gain != nullptr) {
		for (int i = 0; i < gain->geometry()->numDevices(); i++) {
			int deviceId = gain->geometry()->deviceIdForDeviceIdx(i);
			size_t byteSize = NUM_TRANS_IN_UNIT * sizeof(uint16_t);
			memcpy(cursor, &gain->_data[deviceId][0], byteSize);
			cursor += byteSize / sizeof(body[0]);
		}
	}
	return body;
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
				std::unique_ptr<uint8_t[]> body = MakeBody(gain, mod, &body_size);
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
		catch (int errnum) {
			this->Close();
			std::cerr << errnum << "Link closed." << std::endl;
		}
	});
}


void autd::Controller::impl::AppendGain(GainPtr gain) {
	{
		this->FinishLateralModulation();
		gain->SetGeometry(this->_geometry);
		std::unique_lock<std::mutex> lk(_build_mtx);
		_build_q.push(gain);
	}
	_build_cond.notify_all();
}

void autd::Controller::impl::AppendGainSync(autd::GainPtr gain) {
	try {
		this->FinishLateralModulation();
		gain->SetGeometry(this->_geometry);
		if (!gain->built()) gain->build();
		if (gain->_fix) gain->FixImpl();

		size_t body_size = 0;

		std::unique_ptr<uint8_t[]> body = this->MakeBody(gain, nullptr, &body_size);
		if (this->isOpen()) this->_link->Send(body_size, std::move(body));
	}
	catch (int errnum) {
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
				std::unique_ptr<uint8_t[]> body = this->MakeBody(nullptr, mod, &body_size);
				this->_link->Send(body_size, std::move(body));
				std::this_thread::sleep_for(std::chrono::milliseconds(1));
			}
			mod->sent = 0;
		}
	}
	catch (int errnum) {
		this->Close();
		std::cerr << errnum << "Link closed." << std::endl;
	}
}

void autd::Controller::impl::AppendLateralGain(GainPtr gain)
{
	if (this->_plt == nullptr) {
		this->_plt = std::unique_ptr<LateralTimer>(new LateralTimer(this->_link, this->_geometry, this->silentMode));
	}
	gain->SetGeometry(this->_geometry);
	if (!gain->built()) gain->build();
	this->_plt->AppendLateralGain(gain);
}

void autd::Controller::impl::StartLateralModulation(float freq)
{
	if (this->_plt == nullptr || this->_plt->Size() == 0) {
		std::cerr << "Call \"AppendLateralGain\" before start Lateral Modulation" << std::endl;
		return;
	}

	this->FinishLateralModulation();

	int itvl_micro_sec = (int)(1000 * 1000 / freq / this->_plt->Size());
	this->_plt->SetInterval(itvl_micro_sec);
	this->_plt->Start();
	this->LMrunning = true;
}

void autd::Controller::impl::AppendLateralGain(const std::vector<GainPtr> &gain_list)
{
	for (GainPtr g : gain_list) {
		this->AppendLateralGain(g);
	}
}

void autd::Controller::impl::FinishLateralModulation() {
	if (this->LMrunning && this->_plt != nullptr) this->_plt->Stop();
	this->LMrunning = false;
}

void autd::Controller::impl::ResetLateralGain()
{
	this->FinishLateralModulation();
	if (this->_plt != nullptr) this->_plt.reset();
}

void autd::Controller::impl::FlushBuffer() {
	std::unique_lock<std::mutex> lk0(_send_mtx);
	std::unique_lock<std::mutex> lk1(_build_mtx);
	std::queue<GainPtr>().swap(_build_q);
	std::queue<GainPtr>().swap(_send_gain_q);
	std::queue<ModulationPtr>().swap(_send_mod_q);
}

std::unique_ptr<uint8_t[]> autd::Controller::impl::MakeBody(GainPtr gain, ModulationPtr mod, size_t *size) {
	int num_devices = (gain != nullptr) ? gain->geometry()->numDevices() : 0;

	*size = sizeof(RxGlobalHeader) + sizeof(uint16_t)*NUM_TRANS_IN_UNIT*num_devices;
	std::unique_ptr<uint8_t[]> body(new uint8_t[*size]);

	RxGlobalHeader *header = reinterpret_cast<RxGlobalHeader*>(&body[0]);
	header->msg_id = static_cast<uint8_t>(rand() % 256); // NOLINT
	header->control_flags = 0;
	header->mod_size = 0;

	if (this->silentMode) header->control_flags |= SILENT;

	if (mod != nullptr) {
		header->control_flags |= (this->modReset ^= MOD_RESET);

		int mod_size = std::max(0, std::min(static_cast<int>(mod->buffer.size() - mod->sent), MOD_FRAME_SIZE));
		header->mod_size = mod_size;
		if (mod->sent == 0) header->control_flags |= MOD_BEGIN;
		if (mod->loop && mod->sent == 0) header->control_flags |= LOOP_BEGIN;
		if (mod->loop && mod->sent + mod_size >= mod->buffer.size()) header->control_flags |= LOOP_END;
		header->frequency_shift = this->_geometry->_freq_shift;

		memcpy(header->mod, &mod->buffer[mod->sent], mod_size);
		mod->sent += mod_size;
	}

	uint8_t *cursor = &body[0] + sizeof(RxGlobalHeader) / sizeof(body[0]);
	if (gain != nullptr) {
		for (int i = 0; i < gain->geometry()->numDevices(); i++) {
			int deviceId = gain->geometry()->deviceIdForDeviceIdx(i);
			size_t byteSize = NUM_TRANS_IN_UNIT * sizeof(uint16_t);
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
	this->_pimpl = std::unique_ptr<impl>(new impl);
	this->_pimpl->_geometry = GeometryPtr(new Geometry());
	this->_pimpl->frequency_shift = -3;
	this->_pimpl->silentMode = true;
	this->_pimpl->modReset = MOD_RESET;
	this->_pimpl->_plt = nullptr;
	this->_pimpl->LMrunning = false;
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
			this->_pimpl->_link = std::shared_ptr<internal::LocalEthercatLink>(new internal::LocalEthercatLink());
		}
		else {
			this->_pimpl->_link = std::shared_ptr<internal::EthercatLink>(new internal::EthercatLink());
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
	this->_pimpl->AppendGain(gain);
}

void autd::Controller::AppendGainSync(GainPtr gain) {
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
	this->_pimpl->AppendLateralGain(gain);
}

void autd::Controller::AppendLateralGain(const std::vector<GainPtr> &gain_list)
{
	this->_pimpl->AppendLateralGain(gain_list);
}

void autd::Controller::StartLateralModulation(float freq)
{
	this->_pimpl->StartLateralModulation(freq);
}

void autd::Controller::FinishLateralModulation()
{
	this->_pimpl->FinishLateralModulation();
}

void autd::Controller::ResetLateralGain()
{
	this->_pimpl->ResetLateralGain();
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

autd::GeometryPtr autd::Controller::geometry() {
	return this->_pimpl->_geometry;
}

void autd::Controller::SetGeometry(const GeometryPtr &geometry) {
	this->_pimpl->_geometry = geometry;
}

size_t autd::Controller::remainingInBuffer() {
	return this->_pimpl->_send_gain_q.size() + this->_pimpl->_send_mod_q.size() + this->_pimpl->_build_q.size();
}

void autd::Controller::SetSilentMode(bool silent) {
	this->_pimpl->silentMode = silent;
}

bool autd::Controller::silentMode() {
	return this->_pimpl->silentMode;
}
