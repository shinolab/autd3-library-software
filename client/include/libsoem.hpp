/*
 *  libsoem.hpp
 *
 *  Created by Shun Suzuki on 08/22/2019.
 *  Copyright © 2019 Hapis Lab. All rights reserved.
 *
 */

#pragma once

#include <memory>
#include <vector>
#include <string>

constexpr auto TRANS_NUM = 249;
constexpr auto MOD_SIZE = 124;
constexpr auto OUTPUT_FRAME_SIZE = TRANS_NUM * 2 + MOD_SIZE + 4;
constexpr auto INPUT_FRAME_SIZE = 2;

using namespace std;

namespace libsoem {
	class SOEMController
	{
	public:
		SOEMController();
		~SOEMController();

		void Open(const char* ifname, size_t devNum);
		void Send(size_t size, unique_ptr<uint8_t[]> buf);
		void Close();

	private:
		class impl;
		shared_ptr<impl> _pimpl;
	};

	struct EtherCATAdapterInfo {
	public:
		EtherCATAdapterInfo(){};
		EtherCATAdapterInfo(const EtherCATAdapterInfo &info) {
			desc = info.desc;
			name = info.name;
		}
		static vector<EtherCATAdapterInfo> EnumerateAdapters();

		shared_ptr<string> desc;
		shared_ptr<string> name;

	};
}