/*
*
*  Created by Shun Suzuki and Saya Mizutani on 02/07/2018.
*  Copyright © 2018 Hapis Lab. All rights reserved.
*
*/

#include <stdexcept>
#include <future>
#include <iostream>
#include <vector>
#include <Windows.h>
#include <chrono>
#include <cmath>
#include "timer.hpp"

#define TimeScale (1000*1000) //us

using namespace std;

Timer::Timer() {}
Timer::~Timer() {
	this->Stop();
}

void Timer::SetInterval(int interval)
{
	if (interval <= 0) throw new std::runtime_error("Interval must be positive integer.");
	this->_interval = interval;
}

void Timer::Start() {
	this->Stop();
	this->_loop = true;
	this->InitTimer();
}

void Timer::Stop() {
	if (this->_loop) {
		this->_loop = false;
		this->_mainThread.join();
	}
}

void Timer::InitTimer() {
	this->_mainThread = std::thread([&] {Timer::MainLoop(); });
}

inline bool MicroSleep(int micro_sec) {
	LARGE_INTEGER start, end, freq;
	QueryPerformanceFrequency(&freq);

	if (!QueryPerformanceCounter(&start))
		return false;

	while (true)
	{
		if (!QueryPerformanceCounter(&end)) return false;
		auto dur = ((double)(end.QuadPart - start.QuadPart) / freq.QuadPart) * TimeScale;
		if (dur > micro_sec) break;
	}
	return true;
}

void Timer::MainLoop() {

	LARGE_INTEGER start, end, freq;
	LARGE_INTEGER e;
	LARGE_INTEGER fs, fe;
	clock_t sleep_t = 0;
	double delay = 0;
	clock_t t0 = 0;

	QueryPerformanceFrequency(&freq);

	while (this->_loop) {
		QueryPerformanceCounter(&fs);
		delay = 0;
		QueryPerformanceCounter(&start);

		this->Run();

		QueryPerformanceCounter(&end);
		sleep_t = (clock_t)((-(double)(end.QuadPart - start.QuadPart) / freq.QuadPart) *TimeScale + this->_interval - delay);

		delay = 0;

		if (sleep_t > 0) {
			MicroSleep(sleep_t);
			sleep_t = 0;
		}
		if (sleep_t > _interval) break;

		QueryPerformanceCounter(&e);
		delay = ((double)(e.QuadPart - start.QuadPart) / freq.QuadPart) *TimeScale - this->_interval;

	}
	QueryPerformanceCounter(&fe);


}
