//
//  controller.hpp
//  autd3
//
//  Created by Shun Suzuki on 04/11/2018.
//
//

#pragma once

#include <memory>

namespace autd {
	namespace internal {
		class Link;
	}

	enum LinkType {
		ETHERCAT,
		ETHERNET,
		USB,
		SERIAL,
	};

	class Controller;
	class Geometry;

	template <class T>
#if DLL_FOR_CAPI
	static T* CreateHelper() {
		struct impl : T {
			impl() : T() {}
		};
		return new impl;
	}
#else
	static std::shared_ptr<T> CreateHelper() {
		struct impl : T {
			impl() : T() {}
		};
		auto p = std::make_shared<impl>();
		return std::move(p);
	}
#endif
}
