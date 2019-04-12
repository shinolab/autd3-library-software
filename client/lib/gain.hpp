//
//  controller.hpp
//  autd3
//
//  Created by Shun Suzuki on 04/11/2018.
//
//

#pragma once

#include <memory>
#include <mutex>
#include <map>

#include <codeanalysis\warnings.h>
#pragma warning(push)
#pragma warning(disable:ALL_CODE_ANALYSIS_WARNINGS)
#include <Eigen/Core> 
#pragma warning(pop)

#include "core.hpp"
#include "geometry.hpp"

namespace autd {
	class Gain;

#if DLL_FOR_CSHARP
	typedef Gain* GainPtr;
#else
	typedef std::shared_ptr<Gain> GainPtr;
#endif

	class Gain
	{
		friend class Controller;
		friend class Geometry;
		friend class internal::Link;
	protected:
		Gain();
		void FixImpl();

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
		std::mutex _mtx;
		bool _built;
		bool _fix;
		GeometryPtr _geometry;
		std::map<int, std::vector<uint16_t> > _data;
	public:
		static GainPtr Create();
		virtual void build();
		void Fix();
		void SetGeometry(const GeometryPtr &geometry);
		GeometryPtr geometry();
		std::map<int, std::vector<uint16_t> > data();
		bool built();
	};

	typedef Gain NullGain;

	class PlaneWaveGain : public Gain {
	public:
		static GainPtr Create(Eigen::Vector3f direction);
		void build();
	private:
		Eigen::Vector3f _direction;
	};

	class FocalPointGain : public Gain {
	public:
		static GainPtr Create(Eigen::Vector3f point);
		static GainPtr Create(Eigen::Vector3f point, uint8_t amp);
		void build();
	private:
		Eigen::Vector3f _point;
		uint8_t _amp;
	};

	class BesselBeamGain : public Gain {
	public:
		static GainPtr Create(Eigen::Vector3f point, Eigen::Vector3f vec_n, float theta_z);
		void build();
	private:
		Eigen::Vector3f _point;
		Eigen::Vector3f _vec_n;
		float _theta_z;
	};

	class CustomGain : public Gain {
	public:
		static GainPtr Create(uint16_t* data, int dataLength);
		void build();
	private:
		std::vector<uint16_t> _rawdata;
	};

	class GroupedGain : public Gain {
	public:
		static GainPtr Create(std::map<int, autd::GainPtr> gainmap);
		void build();
	private:
		std::map<int, autd::GainPtr> _gainmap;
	};

	class HoloGainSdp : public Gain {
	public:
		static GainPtr Create(Eigen::MatrixX3f foci, Eigen::VectorXf amp);
		virtual void build();
	protected:
		Eigen::MatrixX3f _foci;
		Eigen::VectorXf _amp;
	};

	typedef HoloGainSdp HoloGain;

	class MatlabGain : public Gain {
	public:
		static GainPtr Create(std::string filename, std::string varname);
		void build();
	protected:
		std::string _filename, _varname;
	};

	class TransducerTestGain : public Gain {
	public:
		static GainPtr Create(int transducer_index, int amp, int phase);
		void build();
	protected:
		int _xdcr_idx;
		int _amp, _phase;
	};
}
