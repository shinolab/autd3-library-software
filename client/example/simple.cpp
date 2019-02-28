/*
 *  simple.cpp
 *  simple
 *
 *  Created by Seki Inoue on 5/18/16.
 *  Copyright © 2016 Hapis Lab. All rights reserved.
 *
 */

#include "autd3.hpp"

using namespace std;

int main()
{
	autd::Controller autd;
	autd.Open(autd::LinkType::ETHERCAT);
	if (!autd.isOpen()) return ENXIO;
	
	autd.geometry()->AddDevice(Eigen::Vector3f(0, 0, 0), Eigen::Vector3f(0, 0, 0));
	
	autd.AppendGainSync(autd::FocalPointGain::Create(Eigen::Vector3f(90, 70, 200)));
	autd.AppendModulationSync(autd::SineModulation::Create(150)); // 150Hz AM
	
	std::cout << "press any key to finish..." << std::endl;
	getchar();

	std::cout << "disconnecting..." << std::endl;
	autd.Close();
	return 0;
}