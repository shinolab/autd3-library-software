/*
 *  simple.cpp
 *  simple
 *
 *  Created by Shun Suzuki on 5/18/16.
 *  Copyright © 2019 Hapis Lab. All rights reserved.
 *
 */

#include "autd3.hpp"

using namespace std;

int main()
{
	autd::Controller autd;

	autd.geometry()->AddDevice(Eigen::Vector3f(0, 0, 0), Eigen::Vector3f(0, 0, 0));
	//autd.geometry()->AddDevice(Eigen::Vector3f(0, 0, 0), Eigen::Vector3f(0, 0, 0));

	autd.Open(autd::LinkType::SOEM, "\\Device\\NPF_{B5B631C6-ED16-4780-9C4C-3941AE8120A6}");
	if (!autd.isOpen()) return ENXIO;

	auto gain = autd::FocalPointGain::Create(Eigen::Vector3f(90, 70, 200));

	autd.AppendModulationSync(autd::SineModulation::Create(150)); // 150Hz AM
	autd.AppendGainSync(gain);
	//autd.AppendGainSync(autd::NullGain::Create());

	std::cout << "press any key to finish..." << std::endl;
	getchar();

	autd.Close();
	autd.Open(autd::LinkType::SOEM, "\\Device\\NPF_{B5B631C6-ED16-4780-9C4C-3941AE8120A6}");
	if (!autd.isOpen()) return ENXIO;

	autd.AppendModulationSync(autd::Modulation::Create(0)); // 150Hz AM
	autd.AppendGainSync(autd::NullGain::Create());

	getchar();

	std::cout << "disconnecting..." << std::endl;
	autd.Close();
	return 0;
}