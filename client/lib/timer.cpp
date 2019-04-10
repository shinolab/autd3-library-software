/*
*
*  Created by Shun Suzuki on 04/10/2019.
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

constexpr auto TIME_SCALE = 1000 * 1000L; //us

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

inline void MicroSleep(int micro_sec) {
	LARGE_INTEGER start, end, freq;

	QueryPerformanceFrequency(&freq);

	auto sleep = micro_sec * (freq.QuadPart / TIME_SCALE);

	QueryPerformanceCounter(&start);
	while (true)
	{
		QueryPerformanceCounter(&end);
		if (end.QuadPart - start.QuadPart > sleep) break;
	}
}

void Timer::MainLoop() {
	LARGE_INTEGER start, now, freq;
	auto count = 0xffffffffL;

	QueryPerformanceFrequency(&freq);

	int sleep_t;
	while (this->_loop) {
		if (count > 0xfffffff0) {
			count = 0;
			QueryPerformanceCounter(&start);
		}

		this->Run();

		QueryPerformanceCounter(&now);
		auto elasped = (double)(now.QuadPart - start.QuadPart) / freq.QuadPart * TIME_SCALE;

		sleep_t = (int)(this->_interval * ++count - elasped);
		if (sleep_t > 0) {
			MicroSleep(sleep_t);
		}
		else continue;
	}
}
