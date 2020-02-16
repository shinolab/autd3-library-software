/*
 * File: timer.cpp
 * Project: linux
 * Created Date: 04/09/2019
 * Author: Shun Suzuki
 * -----
 * Last Modified: 16/02/2020
 * Modified By: Shun Suzuki (suzuki@hapis.k.u-tokyo.ac.jp)
 * -----
 * Copyright (c) 2019 Hapis Lab. All rights reserved.
 * 
 */

#include <atomic>
#include <stdexcept>
#include <string>
#include <future>
#include <iostream>
#include <vector>
#include <chrono>
#include <cmath>
#include "../timer.hpp"

#include <signal.h>
#include <time.h>
#include <string.h>

static constexpr auto TIME_SCALE = 1000L; //us to ns

static std::atomic<bool> AUTD3_LIB_TIMER_LOCK(false);

using namespace std;

Timer::Timer(bool highResolusion) noexcept
{
	this->_interval_us = 1;
}
Timer::~Timer() noexcept(false)
{
	this->Stop();
}

void Timer::SetInterval(int interval_us)
{
	if (interval_us <= 0)
		throw new std::runtime_error("Interval must be positive integer.");
	this->_interval_us = interval_us;
}

void Timer::Start(const std::function<void()> &callback)
{
	this->Stop();
	this->cb = callback;
	this->_loop = true;
	this->InitTimer();
}

void Timer::Stop()
{
	if (this->_loop)
	{
		timer_delete(_timer_id);
		this->_loop = false;
	}
}

void Timer::InitTimer()
{
	struct sigaction act;
	struct itimerspec itval;
	struct sigevent se;

	memset(&act, 0, sizeof(struct sigaction));

	act.sa_handler = MainLoop;
	;
	act.sa_flags = SA_RESTART;
	if (sigaction(SIGALRM, &act, NULL) < 0)
	{
		cerr << "Error: sigaction()." << endl;
	}

	itval.it_value.tv_sec = 0;
	itval.it_value.tv_nsec = this->_interval_us * TIME_SCALE;
	itval.it_interval.tv_sec = 0;
	itval.it_interval.tv_nsec = this->_interval_us * TIME_SCALE;

	memset(&se, 0, sizeof(se));
	se.sigev_value.sival_ptr = this;
	se.sigev_notify = SIGEV_THREAD;
	se.sigev_notify_function = Notify;
	se.sigev_notify_attributes = NULL;

	if (timer_create(CLOCK_REALTIME, &se, &_timer_id) < 0)
	{
		cerr << "Error: timer_create." << endl;
	}

	if (timer_settime(_timer_id, 0, &itval, NULL) < 0)
	{
		cerr << "Error: timer_settime." << endl;
	}
}

void Timer::MainLoop(int signum)
{
}

void Timer::Notify(union sigval sv)
{
	bool expected = false;
	if (AUTD3_LIB_TIMER_LOCK.compare_exchange_weak(expected, true))
	{
		(reinterpret_cast<Timer *>(sv.sival_ptr))->cb();
		AUTD3_LIB_TIMER_LOCK.store(false, std::memory_order_release);
	}
}
