/*
*  soem_link.hpp
*  autd3
*
*  Created by Shun Suzuki on 08/23/2019.
*  Copyright © 2019 Hapis Lab. All rights reserved.
*
*/

#pragma once

#include "link.hpp"
#include "libsoem.hpp"

namespace autd {
	namespace internal {
		class SOEMLink : public Link {
		public:
			void Open(std::string location);
			void Close();
			void Send(size_t size, std::unique_ptr<uint8_t[]> buf);
			bool isOpen();
		protected:
			std::unique_ptr<libsoem::SOEMController> _cnt;
			bool _isOpen = false;
			int _devNum = 0;
		};
	}
}
