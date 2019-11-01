/*
 * File: libsoem.hpp
 * Project: include
 * Created Date: 24/08/2019
 * Author: Shun Suzuki
 * -----
 * Last Modified: 01/11/2019
 * Modified By: Shun Suzuki (suzuki@hapis.k.u-tokyo.ac.jp)
 * -----
 * Copyright (c) 2019 Hapis Lab. All rights reserved.
 *
 */

#pragma once

#include <memory>
#include <vector>
#include <string>

using namespace std;

namespace libsoem
{

class SOEMController
{
public:
	SOEMController();
	~SOEMController();

	void Open(const char *ifname, size_t devNum, uint32_t ec_sm3_cyctime, uint32_t ec_sync0_cyctime, size_t header_size, size_t body_size, size_t input_frame_size);
	void Send(size_t size, unique_ptr<uint8_t[]> buf);
	vector<uint16_t> Read(size_t input_frame_idx);
	bool isOpen();
	void Close();

private:
	class impl;
	unique_ptr<impl> _pimpl;
};

struct EtherCATAdapterInfo
{
public:
	EtherCATAdapterInfo(){};
	EtherCATAdapterInfo(const EtherCATAdapterInfo &info)
	{
		desc = info.desc;
		name = info.name;
	}
	static vector<EtherCATAdapterInfo> EnumerateAdapters();

	shared_ptr<string> desc;
	shared_ptr<string> name;
};
} // namespace libsoem
