/*
*
*  Created by Shun Suzuki and Saya Mizutani on 02/07/2018.
*  Copyright © 2018 Hapis Lab. All rights reserved.
*
*/

#pragma once
#include <future>

class Timer {
public:
	Timer();
	~Timer();
	void SetInterval(int interval);
	void Start();
	void Stop();
protected:
	int _interval;
	virtual void Run() = 0;
private:
	std::thread _mainThread;
	bool _loop = false;
	void MainLoop();
	void InitTimer();
};
