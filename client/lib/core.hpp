//
//  controller.hpp
//  autd3
//
//  Created by Shun Suzuki on 04/11/2018.
//
//

#pragma once

namespace autd {
	namespace internal {
		class Link;
	}

	typedef enum {
		ETHERCAT,
		ETHERNET,
		USB,
		SERIAL,
	} LinkType;

	class Controller;
	class Geometry;
}
