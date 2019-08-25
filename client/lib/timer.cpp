﻿/*
*
*  timer.cpp
*  autd3
*
*  Created by Shun Suzuki and Saya Mizutani on 02/07/2018.
*  Copyright © 2018-2019 Hapis Lab. All rights reserved.
*
*/

#include <stdexcept>
#include <string>
#include <future>
#include <iostream>
#include <vector>
#include <Windows.h>
#include <chrono>
#include <cmath>
#include "timer.hpp"

constexpr auto TIME_SCALE = 1000 * 1000L; //us

using namespace std;

Timer::Timer() noexcept {
	this->_interval_us = 1;
}
Timer::~Timer() noexcept(false) {
	this->Stop();
}

void Timer::SetInterval(int interval_us)
{
	if (interval_us <= 0) throw new std::runtime_error("Interval must be positive integer.");
	this->_interval_us = interval_us;
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

inline void MicroSleep(int micro_sec) noexcept {
	LARGE_INTEGER  freq;
	QueryPerformanceFrequency(&freq);

	const auto sleep = micro_sec * (freq.QuadPart / TIME_SCALE);

	LARGE_INTEGER start;
	QueryPerformanceCounter(&start);
	while (true)
	{
		LARGE_INTEGER now;
		QueryPerformanceCounter(&now);
		if (now.QuadPart - start.QuadPart > sleep) break;
	}
}

void Timer::MainLoop() {
	LARGE_INTEGER freq;
	QueryPerformanceFrequency(&freq);

	LARGE_INTEGER start;
	QueryPerformanceCounter(&start);

	auto count = 0xffffffffL;
	int sleep_t = 0;
	while (this->_loop) {
		if (count > 0xfffffff0) {
			count = 0;
			QueryPerformanceCounter(&start);
		}

		this->Run();

		LARGE_INTEGER now;
		QueryPerformanceCounter(&now);
		const auto elasped = static_cast<double>(now.QuadPart - start.QuadPart) / freq.QuadPart * TIME_SCALE;

		sleep_t = static_cast<int>(this->_interval_us * ++count - elasped);
		if (sleep_t > 0) {
			MicroSleep(sleep_t);
		}
		else continue;
	}
}
