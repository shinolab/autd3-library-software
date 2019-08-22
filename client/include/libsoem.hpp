/*
 *  libsoem.hpp
 *
 *  Created by Shun Suzuki on 08/22/2019.
 *  Copyright © 2019 Hapis Lab. All rights reserved.
 *
 */
 
#pragma once

#include <memory>

constexpr auto TRANS_NUM = 249;
constexpr auto MOD_SIZE = 124;
constexpr auto FRAME_SIZE = TRANS_NUM * 2 + MOD_SIZE + 4;

namespace libsoem {
	class SOEMController
	{
	public:
		SOEMController();
		~SOEMController();

		void Open(const char* ifname, size_t devNum);
		void Send(size_t size, std::unique_ptr<uint8_t[]> buf);
		void Close();

	private:
		class impl;
		std::shared_ptr<impl> _pimpl;
	};
}