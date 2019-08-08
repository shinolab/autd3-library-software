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
#include "geometry.hpp"
#include "privdef.hpp"
#include "ethercat_link.hpp"
#include "lateraltimer.hpp"
#include <bitset>
using namespace autd;
using namespace std;

#pragma region Controller::impl
class Controller::impl {
	friend class Controller::lateraltimer;

public:
	GeometryPtr _geometry;
	shared_ptr<internal::Link> _link;
	queue<GainPtr> _build_q;
	queue<GainPtr> _send_gain_q;
	queue<ModulationPtr> _send_mod_q;

	thread _build_thr;
	thread _send_thr;
	condition_variable _build_cond;
	condition_variable _send_cond;
	mutex _build_mtx;
	mutex _send_mtx;

	bool silentMode = true;
	bool lm_silentMode = true;
	uint8_t modReset = MOD_RESET;

	~impl() noexcept(false);
	bool isOpen();
	void Close();

	void InitPipeline();
	void AppendGain(const GainPtr gain);
	void AppendGainSync(const GainPtr gain);
	void AppendModulation(const ModulationPtr mod);
	void AppendModulationSync(const ModulationPtr mod);

	void FlushBuffer();
	unique_ptr<uint8_t[]> MakeBody(GainPtr gain, ModulationPtr mod, size_t* size);
};

Controller::impl::~impl() noexcept(false) {
	if (this_thread::get_id() != this->_build_thr.get_id() && this->_build_thr.joinable()) this->_build_thr.join();
	if (this_thread::get_id() != this->_send_thr.get_id() && this->_send_thr.joinable()) this->_send_thr.join();
}

void Controller::impl::InitPipeline() {
	// pipeline step #1
	this->_build_thr = thread([&] {
		while (this->isOpen()) {
			GainPtr gain = nullptr;
			// wait for gain
			{
				unique_lock<mutex> lk(_build_mtx);
				_build_cond.wait(lk, [&] {
					return _build_q.size() || !this->isOpen();
					});
				if (_build_q.size()) {
					gain = _build_q.front();
					_build_q.pop();
				}
			}

			// build gain
			if (gain != nullptr)
			{
				if (gain->built()) gain->build();
			}
			// pass gain to next pipeline stage
			{
				unique_lock<mutex> lk(_send_mtx);
				_send_gain_q.push(gain);
				_send_cond.notify_all();
			}
		}
		});

	// pipeline step #2
	this->_send_thr = thread([&] {
		try {
			while (this->isOpen()) {
				GainPtr gain = nullptr;
				ModulationPtr mod = nullptr;

				// wait for inputs
				{
					unique_lock<mutex> lk(_send_mtx);
					_send_cond.wait(lk, [&] {
						return _send_gain_q.size() || _send_mod_q.size() || !this->isOpen();
						});
					if (_send_gain_q.size())
						gain = _send_gain_q.front();
					if (_send_mod_q.size())
						mod = _send_mod_q.front();
				}

				size_t body_size = 0;
				auto body = MakeBody(gain, mod, &body_size);
				if (this->_link->isOpen()) this->_link->Send(body_size, move(body));

				// remove elements
				unique_lock<mutex> lk(_send_mtx);
				if (gain != nullptr)
					_send_gain_q.pop();
				if (mod != nullptr && mod->buffer.size() <= mod->sent) {
					mod->sent = 0;
					_send_mod_q.pop();
				}

				this_thread::sleep_for(chrono::milliseconds(1));
			}
		}
		catch (const int errnum) {
			this->Close();
			cerr << errnum << "Link closed." << endl;
		}
		});
}

void Controller::impl::AppendGain(GainPtr gain) {
	{
		gain->SetGeometry(this->_geometry);
		unique_lock<mutex> lk(_build_mtx);
		_build_q.push(gain);
	}
	_build_cond.notify_all();
}

void Controller::impl::AppendGainSync(GainPtr gain) {
	try {
		gain->SetGeometry(this->_geometry);
		if (!gain->built()) gain->build();

		size_t body_size = 0;
		auto body = this->MakeBody(gain, nullptr, &body_size);

		if (this->isOpen()) this->_link->Send(body_size, move(body));
	}
	catch (const int errnum) {
		this->_link->Close();
		cerr << errnum << "Link closed." << endl;
	}
}

void Controller::impl::AppendModulation(ModulationPtr mod) {
	unique_lock<mutex> lk(_send_mtx);
	_send_mod_q.push(mod);
	_send_cond.notify_all();
}

void Controller::impl::AppendModulationSync(ModulationPtr mod) {
	try {
		if (this->isOpen()) {
			while (mod->buffer.size() > mod->sent) {
				size_t body_size = 0;
				auto body = this->MakeBody(nullptr, mod, &body_size);
				this->_link->Send(body_size, move(body));
				this_thread::sleep_for(chrono::milliseconds(1));
			}
			mod->sent = 0;
		}
	}
	catch (const int errnum) {
		this->Close();
		cerr << errnum << "Link closed." << endl;
	}
}

void Controller::impl::FlushBuffer() {
	unique_lock<mutex> lk0(_send_mtx);
	unique_lock<mutex> lk1(_build_mtx);
	queue<GainPtr>().swap(_build_q);
	queue<GainPtr>().swap(_send_gain_q);
	queue<ModulationPtr>().swap(_send_mod_q);
}

unique_ptr<uint8_t[]> Controller::impl::MakeBody(GainPtr gain, ModulationPtr mod, size_t* size) {
	auto num_devices = (gain != nullptr) ? gain->geometry()->numDevices() : 0;

	*size = sizeof(RxGlobalHeader) + sizeof(uint16_t) * NUM_TRANS_IN_UNIT * num_devices;
	auto body = make_unique<uint8_t[]>(*size);

	auto* header = reinterpret_cast<RxGlobalHeader*>(&body[0]);
	header->msg_id = static_cast<uint8_t>(rand() % 256); // NOLINT
	header->control_flags = 0;
	header->mod_size = 0;

	if (this->silentMode) header->control_flags |= SILENT;

	//std::cout << std::bitset<8>(header->control_flags) << std::endl;

	if (mod != nullptr) {
		header->control_flags |= (this->modReset ^= MOD_RESET);

		const uint8_t mod_size = max(0, min(static_cast<int>(mod->buffer.size() - mod->sent), MOD_FRAME_SIZE));
		header->mod_size = mod_size;
		if (mod->sent == 0) header->control_flags |= MOD_BEGIN;
		if (mod->loop && mod->sent == 0) header->control_flags |= LOOP_BEGIN;
		if (mod->loop && mod->sent + mod_size >= mod->buffer.size()) header->control_flags |= LOOP_END;
		header->frequency_shift = this->_geometry->_freq_shift;

		memcpy(header->mod, &mod->buffer[mod->sent], mod_size);
		mod->sent += mod_size;
	}

	auto* cursor = &body[0] + sizeof(RxGlobalHeader) / sizeof(body[0]);
	if (gain != nullptr) {
		for (int i = 0; i < gain->geometry()->numDevices(); i++) {
			auto deviceId = gain->geometry()->deviceIdForDeviceIdx(i);
			auto byteSize = NUM_TRANS_IN_UNIT * sizeof(uint16_t);
			memcpy(cursor, &gain->_data[deviceId].at(0), byteSize);
			cursor += byteSize / sizeof(body[0]);
		}
	}
	return body;
}


bool Controller::impl::isOpen() {
	return this->_link.get() && this->_link->isOpen();
}

void Controller::impl::Close() {
	if (this->isOpen()) {
		this->silentMode = false;
		this->lm_silentMode = false;
		auto nullgain = NullGain::Create();
		this->AppendGainSync(nullgain);
#if DLL_FOR_CSHARP
		delete nullgain;
#endif
		this->_link->Close();
		this->FlushBuffer();
		this->_build_cond.notify_all();
		if (this_thread::get_id() != this->_build_thr.get_id() && this->_build_thr.joinable()) this->_build_thr.join();
		this->_send_cond.notify_all();
		if (this_thread::get_id() != this->_send_thr.get_id() && this->_send_thr.joinable()) this->_send_thr.join();
		this->_link = shared_ptr<internal::EthercatLink>(nullptr);
	}
}

#pragma endregion

#pragma region lateraltimer
class Controller::lateraltimer : public Timer {
	friend class Controller;

public:
	lateraltimer() noexcept;
	void AppendLateralGain(GainPtr gain, const GeometryPtr geometry);
	void AppendLateralGain(const vector<GainPtr>& gain_list, const GeometryPtr geometry);
	void StartLateralModulation(float freq);
	void FinishLateralModulation();
	void ResetLateralGain();
	int Size() noexcept;
protected:
	void Run() override;
private:
	weak_ptr<Controller::impl> _pcnt;
	int _lateral_gain_size;
	int _lateral_gain_idx;
	vector<GainPtr> _lateral_gain;
	bool _running;
};

Controller::lateraltimer::lateraltimer() noexcept {
	this->_lateral_gain_size = 0;
	this->_lateral_gain_idx = 0;
	_running = false;
}

int Controller::lateraltimer::Size() noexcept {
	return this->_lateral_gain_size;
}

void Controller::lateraltimer::Run() {
	try {
		auto gain = this->_lateral_gain.at(this->_lateral_gain_idx);
		this->_lateral_gain_idx = (this->_lateral_gain_idx + 1) % this->_lateral_gain_size;
		{
			auto cnt = this->_pcnt.lock();
			cnt->AppendGainSync(gain);
		}
	}
	catch (const int errnum) {
		{
			auto cnt = this->_pcnt.lock();
			cnt->Close();
		}
		cerr << errnum << "Link closed." << endl;
	}
}

void Controller::lateraltimer::StartLateralModulation(float freq)
{
	if (this->Size() == 0) {
		cerr << "Call \"AppendLateralGain\" before start Lateral Modulation" << endl;
		return;
	}

	this->FinishLateralModulation();

	const auto itvl_us = static_cast<int>(1000 * 1000 / freq / this->Size());
	this->SetInterval(itvl_us);
	this->Start();
	this->_running = true;
}

void Controller::lateraltimer::AppendLateralGain(GainPtr gain, const GeometryPtr geometry) {
	gain->SetGeometry(geometry);
	if (!gain->built()) gain->build();

	this->_lateral_gain_size++;
	this->_lateral_gain_idx = 0;
	this->_lateral_gain.push_back(gain);
}

void Controller::lateraltimer::AppendLateralGain(const vector<GainPtr>& gain_list, const GeometryPtr geometry)
{
	for (auto g : gain_list) {
		this->AppendLateralGain(g, geometry);
	}
}

void Controller::lateraltimer::FinishLateralModulation() {
	if (this->_running) this->Stop();
	this->_running = false;
}

void Controller::lateraltimer::ResetLateralGain()
{
	this->_lateral_gain_size = 0;
	this->_lateral_gain.clear();
}
#pragma endregion

Controller::Controller() noexcept(false) {
	this->_pimpl = make_shared<impl>();
	this->_pimpl->_geometry = Geometry::Create();
	this->_pimpl->silentMode = true;
	this->_pimpl->lm_silentMode = false;
	this->_pimpl->modReset = MOD_RESET;

	this->_ptimer = make_unique<lateraltimer>();
	this->_ptimer->_pcnt = this->_pimpl;
}

Controller::~Controller()  noexcept(false) {
	this->Close();
}

void Controller::Open(LinkType type, string location) {
	this->Close();

	switch (type) {
	case LinkType::ETHERCAT: {
		// TODO(volunteer): a smarter localhost detection
		if (location == "" ||
			location.find("localhost") == 0 ||
			location.find("0.0.0.0") == 0 ||
			location.find("127.0.0.1") == 0) {
			this->_pimpl->_link = make_shared<internal::LocalEthercatLink>();
		}
		else {
			this->_pimpl->_link = make_shared<internal::EthercatLink>();
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

bool Controller::isOpen() {
	return this->_pimpl->isOpen();
}

void Controller::Close() {
	this->_pimpl->Close();
}

void Controller::AppendGain(GainPtr gain) {
	this->_ptimer->FinishLateralModulation();
	this->_pimpl->AppendGain(gain);
}

void Controller::AppendGainSync(GainPtr gain) {
	this->_ptimer->FinishLateralModulation();
	this->_pimpl->AppendGainSync(gain);
}

void Controller::AppendModulation(ModulationPtr modulation) {
	this->_pimpl->AppendModulation(modulation);
}

void Controller::AppendModulationSync(ModulationPtr modulation) {
	this->_pimpl->AppendModulationSync(modulation);
}

void Controller::AppendLateralGain(GainPtr gain)
{
	this->_ptimer->AppendLateralGain(gain, this->geometry());
}

void Controller::AppendLateralGain(const vector<GainPtr>& gain_list)
{
	this->_ptimer->AppendLateralGain(gain_list, this->geometry());
}

void Controller::StartLateralModulation(float freq)
{
	this->_ptimer->StartLateralModulation(freq);
}

void Controller::FinishLateralModulation()
{
	this->AppendGainSync(autd::NullGain::Create());
	this->_ptimer->FinishLateralModulation();
}

void Controller::ResetLateralGain()
{
	this->FinishLateralModulation();
	this->_ptimer->ResetLateralGain();
}

void Controller::Flush() {
	this->_pimpl->FlushBuffer();
}

GeometryPtr Controller::geometry() noexcept {
	return this->_pimpl->_geometry;
}

void Controller::SetGeometry(const GeometryPtr& geometry) noexcept {
	this->_pimpl->_geometry = geometry;
}

size_t Controller::remainingInBuffer() {
	return this->_pimpl->_send_gain_q.size() + this->_pimpl->_send_mod_q.size() + this->_pimpl->_build_q.size();
}

void Controller::SetSilentMode(bool silent) noexcept {
	this->_pimpl->silentMode = silent;
}

void Controller::SetLMSilentMode(bool silent) noexcept {
	this->_pimpl->lm_silentMode = silent;
}

bool Controller::silentMode() noexcept {
	return this->_pimpl->silentMode;
}
