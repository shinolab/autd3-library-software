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
			bool _isInit;
			bool _isOpen;
			int _devNum;
			std::string _ifname;
		};
	}
}
