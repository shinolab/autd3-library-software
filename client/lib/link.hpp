﻿/*
 *  link.hpp
 *  autd3
 *
 *  Created by Seki Inoue on 6/1/16.
 *  Modified by Shun Suzuki on 02/07/2018.
 *  Copyright © 2016-2019 Hapis Lab. All rights reserved.
 *
 */

#ifndef link_hpp
#define link_hpp

#include <stdio.h>
#include <string>
#include <vector>
#include "gain.hpp"
#include "modulation.hpp"

namespace autd {
	namespace internal {
		class Link {
		public:
			virtual void Open(std::string location) = 0;
			virtual void Close() = 0;
			virtual void Send(size_t size, std::unique_ptr<uint8_t[]> buf) = 0;
			virtual bool isOpen() = 0;
		};
	}
}

#endif /* link_hpp */
