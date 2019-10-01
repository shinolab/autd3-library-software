/*
 * File: holo_gain.cpp
 * Project: lib
 * Created Date: 06/07/2016
 * Author: Seki Inoue
 * -----
 * Last Modified: 05/09/2019
 * Modified By: Shun Suzuki (suzuki@hapis.k.u-tokyo.ac.jp)
 * -----
 * Copyright (c) 2016-2019 Hapis Lab. All rights reserved.
 * 
 */

#include <iostream>
#include <string>
#include <map>
#include <complex>
#include <vector>
#include <random>

#if WIN32
#include <codeanalysis\warnings.h>
#pragma warning(push)
#pragma warning(disable \
				: ALL_CODE_ANALYSIS_WARNINGS)
#endif
#include <Eigen/Eigen>
#if WIN32
#pragma warning(pop)
#endif

#include "privdef.hpp"
#include "autd3.hpp"
#include "controller.hpp"
#include "gain.hpp"
#define _USE_MATH_DEFINES
#include <math.h>

constexpr auto REPEAT_SDP = 10;
constexpr auto LAMBDA_SDP = 0.8;

using namespace std;
using namespace autd;
using namespace Eigen;

namespace hologainimpl
{
// TODO: use cache and interpolate. don't use exp
complex<float> transfer(Vector3f trans_pos, Vector3f trans_norm, Vector3f target_pos)
{
	const auto diff = trans_pos - target_pos;
	const auto dist = diff.norm();
	const auto cos = diff.dot(trans_norm) / dist / trans_norm.norm();

	//1.11   & 1.06  &  0.24 &  -0.12  &  0.035
	const auto directivity = 1.11f * sqrt((2 * 0 + 1) / (4 * M_PIf)) * 1 + 1.06f * sqrt((2 * 1 + 1) / (4 * M_PIf)) * cos + 0.24f * sqrt((2 * 2 + 1) / (4 * M_PIf)) / 2 * (3 * cos * cos - 1) - 0.12f * sqrt((2 * 3 + 1) / (4 * M_PIf)) / 2 * cos * (5 * cos * cos - 3) + 0.035f * sqrt((2 * 4 + 1) / (4 * M_PIf)) / 8 * (35 * cos * cos * cos * cos - 30 * cos * cos + 3);

	auto g = directivity / dist * exp(complex<float>(-dist * 1.15e-4f, -2 * M_PIf / ULTRASOUND_WAVELENGTH * dist));
	return g;
}

void removeRow(MatrixXcf &matrix, size_t rowToRemove)
{
	const auto numRows = static_cast<size_t>(matrix.rows()) - 1;
	const auto numCols = static_cast<size_t>(matrix.cols());

	if (rowToRemove < numRows)
		matrix.block(rowToRemove, 0, numRows - rowToRemove, numCols) = matrix.block(rowToRemove + 1, 0, numRows - rowToRemove, numCols);

	matrix.conservativeResize(numRows, numCols);
}

void removeColumn(MatrixXcf &matrix, size_t colToRemove)
{
	const auto numRows = static_cast<size_t>(matrix.rows());
	const auto numCols = static_cast<size_t>(matrix.cols()) - 1;

	if (colToRemove < numCols)
		matrix.block(0, colToRemove, numRows, numCols - colToRemove) = matrix.block(0, colToRemove + 1, numRows, numCols - colToRemove);

	matrix.conservativeResize(numRows, numCols);
}
} // namespace hologainimpl

autd::GainPtr autd::HoloGainSdp::Create(MatrixX3f foci, VectorXf amp)
{
	auto ptr = CreateHelper<HoloGainSdp>();
	ptr->_foci = foci;
	ptr->_amp = amp;
	return ptr;
}

void autd::HoloGainSdp::build()
{
	if (this->built())
		return;
	auto geo = this->geometry();
	if (geo == nullptr)
	{
		throw runtime_error("Geometry is required to build Gain");
	}

	const auto alpha = 1e-3f;

	const size_t M = _foci.rows();
	const auto N = static_cast<int>(geo->numTransducers());

	MatrixXcf P = MatrixXcf::Zero(M, M);
	VectorXcf p = VectorXcf::Zero(M);
	MatrixXcf B = MatrixXcf(M, N);
	VectorXcf q = VectorXcf(N);

	std::random_device rnd;
	std::mt19937 mt(rnd());
	std::uniform_real_distribution<float> range(0, 1);

	for (int i = 0; i < M; i++)
	{
		p(i) = _amp(i) * exp(complex<float>(0.0f, 2.0f * M_PIf * range(mt)));

		P(i, i) = _amp(i);

		const auto tp = _foci.row(i);
		for (int j = 0; j < N; j++)
		{
			B(i, j) = hologainimpl::transfer(
				geo->position(j),
				geo->direction(j),
				tp);
		}
	}

	JacobiSVD<MatrixXcf> svd(B, ComputeThinU | ComputeThinV);
	JacobiSVD<MatrixXcf>::SingularValuesType singularValues_inv = svd.singularValues();
	for (long i = 0; i < singularValues_inv.size(); ++i)
	{
		singularValues_inv(i) = singularValues_inv(i) / (singularValues_inv(i) * singularValues_inv(i) + alpha * alpha);
	}
	MatrixXcf pinvB = (svd.matrixV() * singularValues_inv.asDiagonal() * svd.matrixU().adjoint());

	MatrixXcf MM = P * (MatrixXcf::Identity(M, M) - B * pinvB) * P;
	MatrixXcf X = MatrixXcf::Identity(M, M);
	for (int i = 0; i < M * REPEAT_SDP; i++)
	{
		auto ii = static_cast<size_t>(M * static_cast<double>(range(mt)));

		auto Xc = X;
		hologainimpl::removeRow(Xc, ii);
		hologainimpl::removeColumn(Xc, ii);
		VectorXcf MMc = MM.col(ii);
		MMc.block(ii, 0, MMc.rows() - 1 - ii, 1) = MMc.block(ii + 1, 0, MMc.rows() - 1 - ii, 1);
		MMc.conservativeResize(MMc.rows() - 1, 1);

		VectorXcf x = Xc * MMc;
		complex<float> gamma = x.adjoint() * MMc;
		if (gamma.real() > 1e-7)
		{
			x = -x * sqrt(LAMBDA_SDP / gamma.real());
			X.block(ii, 0, 1, ii) = x.block(0, 0, ii, 1).adjoint().eval();
			X.block(ii, ii + 1, 1, M - ii - 1) = x.block(ii, 0, M - 1 - ii, 1).adjoint().eval();
			X.block(0, ii, ii, 1) = x.block(0, 0, ii, 1).eval();
			X.block(ii + 1, ii, M - ii - 1, 1) = x.block(ii, 0, M - 1 - ii, 1).eval();
		}
		else
		{
			X.block(ii, 0, 1, ii) = VectorXcf::Zero(ii).adjoint();
			X.block(ii, ii + 1, 1, M - ii - 1) = VectorXcf::Zero(M - ii - 1).adjoint();
			X.block(0, ii, ii, 1) = VectorXcf::Zero(ii);
			X.block(ii + 1, ii, M - ii - 1, 1) = VectorXcf::Zero(M - ii - 1);
		}
	}

	ComplexEigenSolver<MatrixXcf> ces(X);
	VectorXcf evs = ces.eigenvalues();
	float abseiv = 0;
	int idx = 0;
	for (int j = 0; j < evs.rows(); j++)
	{
		const auto eiv = abs(evs(j));
		if (abseiv < eiv)
		{
			abseiv = eiv;
			idx = j;
		}
	}

	VectorXcf u = ces.eigenvectors().col(idx);
	q = pinvB * P * u;

	this->_data.clear();
	const int ndevice = geo->numDevices();
	for (int i = 0; i < ndevice; i++)
	{
		this->_data[geo->deviceIdForDeviceIdx(i)].resize(NUM_TRANS_IN_UNIT);
	}

	//auto maxCoeff = sqrt(q.cwiseAbs2().maxCoeff());
	for (int j = 0; j < N; j++)
	{
		const auto famp = 1.0f; //abs(q(j)) / maxCoeff;
		const auto fphase = arg(q(j)) / (2 * M_PIf) + 0.5f;
		const auto amp = static_cast<uint8_t>(famp * 255);
		const auto phase = static_cast<uint8_t>((1 - fphase) * 255);
		uint8_t D, S;
		SignalDesign(amp, phase, D, S);
		this->_data[geo->deviceIdForTransIdx(j)].at(j % NUM_TRANS_IN_UNIT) = (static_cast<uint16_t>(D) << 8) + S;
	}
}
