/*
*
*  timer.hpp
*  autd3
*
*  Created by Shun Suzuki and Saya Mizutani on 02/07/2018.
*  Copyright © 2018-2019 Hapis Lab. All rights reserved.
*
*/

#pragma once
#include <future>

#if WIN32
#else
#include <time.h>
#include <signal.h>
#endif

class Timer {
public:
	Timer() noexcept;
	~Timer() noexcept(false);
	void SetInterval(int interval);
	void Start();
	void Stop();

	Timer(const Timer&) = default;
	Timer(Timer&&) = default;
	Timer& operator=(const Timer&) = default;
	Timer& operator=(Timer&&) = default;
protected:
	int _interval_us;
#if WIN32
#else
    timer_t _timer_id;
#endif
	virtual void Run() = 0;
private:
	std::thread _mainThread;
	bool _loop = false;
#if WIN32
	void MainLoop();
#else
	static void MainLoop(int signum);
	static void Notify(union sigval sv);
#endif
	void InitTimer();
};
