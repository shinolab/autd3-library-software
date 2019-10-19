/*
 * File: soem_link.hpp
 * Project: lib
 * Created Date: 24/08/2019
 * Author: Shun Suzuki
 * -----
 * Last Modified: 19/10/2019
 * Modified By: Shun Suzuki (suzuki@hapis.k.u-tokyo.ac.jp)
 * -----
 * Copyright (c) 2019 Hapis Lab. All rights reserved.
 * 
 */

#pragma once

#include "link.hpp"
#include "libsoem.hpp"

namespace autd
{
namespace internal
{
class SOEMLink : public Link
{
public:
	void Open(std::string location);
	void Close();
	void Send(size_t size, std::unique_ptr<uint8_t[]> buf);
	bool isOpen();
	void CalibrateModulation();

protected:
	std::unique_ptr<libsoem::SOEMController> _cnt;
	bool _isOpen = false;
	int _devNum = 0;
	std::string _ifname = "";
	uint8_t _id = 0;
};
} // namespace internal
} // namespace autd
