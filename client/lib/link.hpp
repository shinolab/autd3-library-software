//
//  link.hpp
//  autd3
//
//  Created by Seki Inoue on 6/1/16.
//  Changed by Shun Suzuki on 02/07/2018.
//
//

#ifndef link_hpp
#define link_hpp

#include <stdio.h>
#include <string>
#include <vector>
#include "autd3.hpp"

#pragma GCC visibility push(hidden)

namespace autd {
    namespace internal {
        class Link {
        protected:
            virtual std::vector<uint16_t> &accessGainData(GainPtr gain, const int deviceId);
            virtual uint32_t &accessSent(ModulationPtr mod);
        public:
            virtual void Open(std::string location) = 0;
            virtual void Close() = 0;
            virtual void Send(size_t size, std::unique_ptr<uint8_t[]> buf) = 0;
            virtual bool isOpen() = 0;
        };
    }
}

#pragma GCC visibility pop

#endif /* link_hpp */
