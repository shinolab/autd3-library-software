/*
 * File: simple_soem.cpp
 * Project: example_soem
 * Created Date: 24/08/2019
 * Author: Shun Suzuki
 * -----
 * Last Modified: 04/09/2019
 * Modified By: Shun Suzuki (suzuki@hapis.k.u-tokyo.ac.jp)
 * -----
 * Copyright (c) 2019 Hapis Lab. All rights reserved.
 * 
 */

#include "autd3.hpp"

using namespace std;

string GetAdapterName()
{
	int size;
	auto adapters = autd::Controller::EnumerateAdapters(size);
	for (auto i = 0; i < size; i++)
	{
		auto adapter = adapters[i];
		cout << "[" << i << "]: " << *adapter.first << ", " << *adapter.second << endl;
	}

	int index;
	cout << "Choose number: ";
	cin >> index;
	cin.ignore();

	return *adapters[index].second;
}

int main()
{
	autd::Controller autd;

	// AddDevice() must be called before Open(), and be called as many times as for the number of AUTDs connected.
	autd.geometry()->AddDevice(Eigen::Vector3f(0, 0, 0), Eigen::Vector3f(0, 0, 0));
	//autd.geometry()->AddDevice(Eigen::Vector3f(0, 151.4f, 0), Eigen::Vector3f(0, 0, 0));

	auto ifname = GetAdapterName();
	autd.Open(autd::LinkType::SOEM, ifname);
	// If you have already recognized the EtherCAT adapter name, you can write it directly like below.
	//autd.Open(autd::LinkType::SOEM, "\\Device\\NPF_{B5B631C6-ED16-4780-9C4C-3941AE8120A6}");

	if (!autd.isOpen())
		return ENXIO;

	auto gain = autd::FocalPointGain::Create(Eigen::Vector3f(90, 70, 150));

	autd.AppendModulationSync(autd::SineModulation::Create(150)); // 150Hz AM
	autd.AppendGainSync(gain);

	std::cout << "press any key to finish..." << std::endl;
	getchar();

	std::cout << "disconnecting..." << std::endl;
	autd.Close();
	return 0;
}
