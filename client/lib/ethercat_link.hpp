//
//  ethercat_link.hpp
//  autd3
//
//  Created by Seki Inoue on 6/1/16.
//
//

#ifndef ethercat_link_hpp
#define ethercat_link_hpp

#include <stdio.h>
#include <string>
#include <AdsLib.h>
#include "link.hpp"

#ifdef _WINDOWS
#define NOMINMAX
#include <Windows.h>
#include <winnt.h>
#else
typedef void* HMODULE;
#endif

/* The classes below are not exported */
#pragma GCC visibility push(hidden)

namespace autd {
    namespace internal {
        class EthercatLink : public Link {
        public:
            virtual void Open(std::string location);
            virtual void Open(std::string ams_net_id, std::string ipv4addr);
            virtual void Close();
            virtual void Send(size_t size, std::unique_ptr<uint8_t[]> buf);
            bool isOpen();
        protected:
            long _port;
            AmsNetId _netId;
        };
        
        class LocalEthercatLink : public EthercatLink {
        public:
			void Open(std::string location = "");
			void Close();
			void Send(size_t size, std::unique_ptr<uint8_t[]> buf);
		private:
			HMODULE lib;
        };
    }
}

#pragma GCC visibility pop

#endif /* ethercat_link_hpp */
