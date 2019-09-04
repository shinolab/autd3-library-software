/*
 * File: simple.cpp
 * Project: example
 * Created Date: 18/05/2016
 * Author: Seki Inoue
 * -----
 * Last Modified: 04/09/2019
 * Modified By: Shun Suzuki (suzuki@hapis.k.u-tokyo.ac.jp)
 * -----
 * Copyright (c) 2016-2019 Hapis Lab. All rights reserved.
 * 
 */

#include "autd3.hpp"

using namespace std;

int main()
{
	autd::Controller autd;
	autd.Open(autd::LinkType::ETHERCAT);
	if (!autd.isOpen())
		return ENXIO;

	autd.geometry()->AddDevice(Eigen::Vector3f(0, 0, 0), Eigen::Vector3f(0, 0, 0));

	auto gain = autd::FocalPointGain::Create(Eigen::Vector3f(90, 70, 200));

	autd.AppendGainSync(gain);
	autd.AppendModulationSync(autd::SineModulation::Create(150)); // 150Hz AM

	std::cout << "press any key to finish..." << std::endl;
	getchar();

	std::cout << "disconnecting..." << std::endl;
	autd.Close();
	return 0;
}