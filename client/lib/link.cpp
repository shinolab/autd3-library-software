//
//  link.cpp
//  autd3
//
//  Created by Seki Inoue on 6/17/16.
//  Modified by Shun Suzuki on 02/07/2018.
//
//

#include <stdio.h>
#include "link.hpp"

std::vector<uint16_t>& autd::internal::Link::accessGainData(Gain& gain, const int deviceId) {
    return gain._data[deviceId];
}

size_t& autd::internal::Link::accessSent(Modulation& mod) noexcept {
    return mod.sent;
}