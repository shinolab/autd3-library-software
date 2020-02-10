/*
 * File: timer.cpp
 * Project: macosx
 * Created Date: 04/09/2019
 * Author: Shun Suzuki
 * -----
 * Last Modified: 10/02/2020
 * Modified By: Shun Suzuki (suzuki@hapis.k.u-tokyo.ac.jp)
 * -----
 * Copyright (c) 2019 Hapis Lab. All rights reserved.
 * 
 */

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

constexpr auto TIME_SCALE = 1000 * 1000L; //us

using namespace std;

Timer::Timer() noexcept
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
		dispatch_source_cancel(_timer);
		this->_loop = false;
	}
}

void Timer::InitTimer()
{
	_queue = dispatch_queue_create("timerQueue", 0);

	_timer = dispatch_source_create(DISPATCH_SOURCE_TYPE_TIMER, 0, 0, _queue);
	dispatch_source_set_event_handler(_timer, ^{
	  MainLoop(this);
	});

	dispatch_source_set_cancel_handler(_timer, ^{
	  dispatch_release(_timer);
	  dispatch_release(_queue);
	});

	dispatch_time_t start = dispatch_time(DISPATCH_TIME_NOW, 0);
	dispatch_source_set_timer(_timer, start, 1000 * 1000, 0);
	dispatch_resume(_timer);
}

void Timer::MainLoop(Timer *ptr)
{
	ptr->cb();
}
