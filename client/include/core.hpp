/*
 * File: core.hpp
 * Project: include
 * Created Date: 11/04/2018
 * Author: Shun Suzuki
 * -----
 * Last Modified: 04/09/2019
 * Modified By: Shun Suzuki (suzuki@hapis.k.u-tokyo.ac.jp)
 * -----
 * Copyright (c) 2019 Hapis Lab. All rights reserved.
 * 
 */

#pragma once

#include <memory>
#include <vector>
#include <string>

namespace autd
{
namespace internal
{
class Link;
}

enum LinkType
{
	ETHERCAT,
	ETHERNET,
	USB,
	SERIAL,
	SOEM
};

class Controller;
class Geometry;

template <class T>
#if DLL_FOR_CAPI
static T *CreateHelper()
{
	struct impl : T
	{
		impl() : T() {}
	};
	return new impl;
}
#else
static std::shared_ptr<T> CreateHelper()
{
	struct impl : T
	{
		impl() : T() {}
	};
	auto p = std::make_shared<impl>();
	return std::move(p);
}
#endif

static std::vector<std::string> split(const std::string &s, char delim)
{
	std::vector<std::string> tokens;
	std::string token;
	for (char ch : s)
	{
		if (ch == delim)
		{
			if (!token.empty())
				tokens.push_back(token);
			token.clear();
		}
		else
			token += ch;
	}
	if (!token.empty())
		tokens.push_back(token);
	return tokens;
}
} // namespace autd
