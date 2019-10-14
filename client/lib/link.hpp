/*
 * File: link.hpp
 * Project: lib
 * Created Date: 01/06/2016
 * Author: Seki Inoue
 * -----
 * Last Modified: 14/10/2019
 * Modified By: Shun Suzuki (suzuki@hapis.k.u-tokyo.ac.jp)
 * -----
 * Copyright (c) 2016-2019 Hapis Lab. All rights reserved.
 * 
 */

#ifndef link_hpp
#define link_hpp

#include <stdio.h>
#include <string>
#include <vector>
#include "gain.hpp"
#include "modulation.hpp"

namespace autd
{
namespace internal
{
class Link
{
public:
	virtual void Open(std::string location) = 0;
	virtual void Close() = 0;
	virtual void Send(size_t size, std::unique_ptr<uint8_t[]> buf) = 0;
	virtual bool isOpen() = 0;
};
} // namespace internal
} // namespace autd

#endif /* link_hpp */
