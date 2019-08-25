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
	virtual void Run() = 0;
private:
	std::thread _mainThread;
	bool _loop = false;
	void MainLoop();
	void InitTimer();
};
