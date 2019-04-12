//
//  controller.hpp
//  autd3
//
//  Created by Shun Suzuki on 04/11/2018.
//
//

#pragma once

#include <memory>
#include <vector>
#include "core.hpp"

namespace autd {
	class Modulation;

#if DLL_FOR_CSHARP
	typedef Modulation* ModulationPtr;
#else
	typedef std::shared_ptr<Modulation> ModulationPtr;
#endif

	class Modulation
	{
		friend class Controller;
		friend class internal::Link;
	public:
		static ModulationPtr Create();
		static ModulationPtr Create(uint8_t amp);
		constexpr float samplingFrequency();
		bool loop;
		std::vector<uint8_t> buffer;
	protected:
		Modulation();
		template <class T>
#if DLL_FOR_CSHARP
		static T* CreateHelper() {
			return new T;
		}
#else
		static std::shared_ptr<T> CreateHelper() {
			return std::shared_ptr<T>(new T());
		}
#endif
	private:
		uint32_t sent;
	};

	class SineModulation : public Modulation
	{
	public:
		static ModulationPtr Create(float freq, float amp = 1.0f, float offset = 0.5f);
	};

	class SawModulation : public Modulation
	{
	public:
		static ModulationPtr Create(float freq);
	};

	class RawPCMModulation : public Modulation
	{
	public:
		static ModulationPtr Create(std::string filename, float samplingFreq = 0.0f);
	};
}