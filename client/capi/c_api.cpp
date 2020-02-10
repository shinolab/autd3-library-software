/*
 * File: c_api.cpp
 * Project: capi
 * Created Date: 02/07/2018
 * Author: Shun Suzuki
 * -----
 * Last Modified: 10/02/2020
 * Modified By: Shun Suzuki (suzuki@hapis.k.u-tokyo.ac.jp)
 * -----
 * Copyright (c) 2019 Hapis Lab. All rights reserved.
 *
 */

#include "autd3.hpp"
#include "autd3_c_api.h"
#include <errno.h>
#include <windows.h>
#include <codeanalysis\warnings.h>
#pragma warning(push)
#pragma warning(disable \
				: ALL_CODE_ANALYSIS_WARNINGS)
#include <Eigen/Geometry>
#pragma warning(pop)

using namespace autd;
using namespace Eigen;

#pragma region Controller
void AUTDCreateController(AUTDControllerHandle *out)
{
	auto *cnt = new Controller;
	*out = cnt;
}
int AUTDOpenController(AUTDControllerHandle handle, int linkType, const char *location)
{
	auto *cnt = static_cast<Controller *>(handle);
	cnt->Open(static_cast<LinkType>(linkType), std::string(location));
	if (!cnt->isOpen())
		return ENXIO;
	return 0;
}
int AUTDGetAdapterPointer(void **out)
{
	int size;
	auto adapters = autd::Controller::EnumerateAdapters(size);
	*out = adapters;
	return size;
}
void AUTDGetAdapter(void *p_adapter, int index, char *desc, char *name)
{
	auto *adapters = static_cast<std::pair<std::string, std::string> *>(p_adapter);
	auto desc_ = adapters[index].first;
	auto name_ = adapters[index].second;
	std::char_traits<char>::copy(desc, desc_.c_str(), desc_.size() + 1);
	std::char_traits<char>::copy(name, name_.c_str(), name_.size() + 1);
}
void AUTDFreeAdapterPointer(void *p_adapter)
{
	auto *adapters = static_cast<std::pair<std::string, std::string> *>(p_adapter);
	delete[] adapters;
}
int AUTDAddDevice(AUTDControllerHandle handle, float x, float y, float z, float rz1, float ry, float rz2, int groupId)
{
	auto *cnt = static_cast<Controller *>(handle);
	return cnt->geometry()->AddDevice(Vector3f(x, y, z), Vector3f(rz1, ry, rz2), groupId);
}
int AUTDAddDeviceQuaternion(AUTDControllerHandle handle, float x, float y, float z, float qua_w, float qua_x, float qua_y, float qua_z, int groupId)
{
	auto *cnt = static_cast<Controller *>(handle);
	return cnt->geometry()->AddDeviceQuaternion(Vector3f(x, y, z), Quaternionf(qua_w, qua_x, qua_y, qua_z), groupId);
}
void AUTDDelDevice(AUTDControllerHandle handle, int devId)
{
	auto *cnt = static_cast<Controller *>(handle);
	cnt->geometry()->DelDevice(devId);
}
void AUTDCloseController(AUTDControllerHandle handle)
{
	auto *cnt = static_cast<Controller *>(handle);
	cnt->Close();
}
void AUTDFreeController(AUTDControllerHandle handle)
{
	auto *cnt = static_cast<Controller *>(handle);
	delete cnt;
}
void AUTDSetSilentMode(AUTDControllerHandle handle, bool mode)
{
	auto *cnt = static_cast<Controller *>(handle);
	cnt->SetSilentMode(mode);
}
void AUTDCalibrateModulation(AUTDControllerHandle handle)
{
	auto *cnt = static_cast<Controller *>(handle);
	cnt->CalibrateModulation();
}
#pragma endregion

#pragma region Property
bool AUTDIsOpen(AUTDControllerHandle handle)
{
	Controller *cnt = static_cast<Controller *>(handle);
	return cnt->isOpen();
}
bool AUTDIsSilentMode(AUTDControllerHandle handle)
{
	Controller *cnt = static_cast<Controller *>(handle);
	return cnt->silentMode();
}
int AUTDNumDevices(AUTDControllerHandle handle)
{
	Controller *cnt = static_cast<Controller *>(handle);
	return cnt->geometry()->numDevices();
}
int AUTDNumTransducers(AUTDControllerHandle handle)
{
	Controller *cnt = static_cast<Controller *>(handle);
	return cnt->geometry()->numTransducers();
}
float AUTDFrequency(AUTDControllerHandle handle)
{
	Controller *cnt = static_cast<Controller *>(handle);
	return cnt->geometry()->frequency();
}
size_t AUTDRemainingInBuffer(AUTDControllerHandle handle)
{
	Controller *cnt = static_cast<Controller *>(handle);
	return cnt->remainingInBuffer();
}
#pragma endregion

#pragma region Gain
void AUTDFocalPointGain(AUTDGainPtr *gain, float x, float y, float z, uint8_t amp)
{
	auto *g = FocalPointGain::Create(Vector3f(x, y, z), amp);
	*gain = g;
}
void AUTDGroupedGain(AUTDGainPtr *gain, int *groupIDs, AUTDGainPtr *gains, int size)
{
	std::map<int, GainPtr> gainmap;

	for (int i = 0; i < size; i++)
	{
		auto id = groupIDs[i];
		auto gain_id = gains[i];
		auto *g = static_cast<Gain *>(gain_id);
		gainmap[id] = g;
	}

	auto *ggain = GroupedGain::Create(gainmap);

	*gain = ggain;
}
void AUTDBesselBeamGain(AUTDGainPtr *gain, float x, float y, float z, float n_x, float n_y, float n_z, float theta_z)
{
	auto *g = BesselBeamGain::Create(Vector3f(x, y, z), Vector3f(n_x, n_y, n_z), theta_z);
	*gain = g;
}
void AUTDPlaneWaveGain(AUTDGainPtr *gain, float n_x, float n_y, float n_z)
{
	auto *g = PlaneWaveGain::Create(Vector3f(n_x, n_y, n_z));
	*gain = g;
}
void AUTDMatlabGain(AUTDGainPtr *gain, const char *filename, const char *varname)
{
	auto *g = MatlabGain::Create(std::string(filename), std::string(varname));
	*gain = g;
}
void AUTDCustomGain(AUTDGainPtr *gain, uint16_t *data, int dataLength)
{
	auto *g = CustomGain::Create(data, dataLength);
	*gain = g;
}
void AUTDHoloGain(AUTDGainPtr *gain, float *points, float *amps, int size)
{

	MatrixX3f holo(size, 3);
	VectorXf amp(size);
	for (int i = 0; i < size; i++)
	{
		holo(i, 0) = points[3 * i];
		holo(i, 1) = points[3 * i + 1];
		holo(i, 2) = points[3 * i + 2];
		amp(i) = amps[i];
	}

	auto *g = HoloGainSdp::Create(holo, amp);
	*gain = g;
}
void AUTDTransducerTestGain(AUTDGainPtr *gain, int idx, int amp, int phase)
{
	auto *g = TransducerTestGain::Create(idx, amp, phase);
	*gain = g;
}
void AUTDNullGain(AUTDGainPtr *gain)
{
	auto *g = NullGain::Create();
	*gain = g;
}
void AUTDDeleteGain(AUTDGainPtr gain)
{
	auto *g = static_cast<Gain *>(gain);
	delete g;
}
#pragma endregion

#pragma region Modulation
void AUTDModulation(AUTDModulationPtr *mod, uint8_t amp)
{
	auto *m = Modulation::Create(amp);
	*mod = m;
}
void AUTDRawPCMModulation(AUTDModulationPtr *mod, const char *filename, float sampFreq)
{
	auto *m = RawPCMModulation::Create(std::string(filename), sampFreq);
	*mod = m;
}
void AUTDSawModulation(AUTDModulationPtr *mod, int freq)
{
	auto *m = SawModulation::Create(freq);
	*mod = m;
}
void AUTDSineModulation(AUTDModulationPtr *mod, int freq, float amp, float offset)
{
	auto *m = SineModulation::Create(freq, amp, offset);
	*mod = m;
}
void AUTDDeleteModulation(AUTDModulationPtr mod)
{
	auto *m = static_cast<Modulation *>(mod);
	delete m;
}
#pragma endregion

#pragma region LowLevelInterface
void AUTDAppendGain(AUTDControllerHandle handle, AUTDGainPtr gain)
{
	auto *cnt = static_cast<Controller *>(handle);
	auto *g = static_cast<Gain *>(gain);
	cnt->AppendGain(g);
}

void AUTDAppendGainSync(AUTDControllerHandle handle, AUTDGainPtr gain)
{
	auto *cnt = static_cast<Controller *>(handle);
	auto *g = static_cast<Gain *>(gain);
	cnt->AppendGainSync(g);
}
void AUTDAppendModulation(AUTDControllerHandle handle, AUTDModulationPtr mod)
{
	auto *cnt = static_cast<Controller *>(handle);
	auto *m = static_cast<Modulation *>(mod);
	cnt->AppendModulation(m);
}
void AUTDAppendModulationSync(AUTDControllerHandle handle, AUTDModulationPtr mod)
{
	auto *cnt = static_cast<Controller *>(handle);
	auto *m = static_cast<Modulation *>(mod);
	cnt->AppendModulationSync(m);
}
void AUTDAppendSTMGain(AUTDControllerHandle handle, AUTDGainPtr gain)
{
	auto *cnt = static_cast<Controller *>(handle);
	auto *g = static_cast<Gain *>(gain);
	cnt->AppendSTMGain(g);
}
void AUTDStartSTModulation(AUTDControllerHandle handle, float freq)
{
	auto *cnt = static_cast<Controller *>(handle);
	cnt->StartSTModulation(freq);
}
void AUTDStopSTModulation(AUTDControllerHandle handle)
{
	auto *cnt = static_cast<Controller *>(handle);
	cnt->StopSTModulation();
}
void AUTDFinishSTModulation(AUTDControllerHandle handle)
{
	auto *cnt = static_cast<Controller *>(handle);
	cnt->FinishSTModulation();
}
void AUTDSetGain(AUTDControllerHandle handle, int deviceIndex, int transIndex, int amp, int phase)
{
	auto *cnt = static_cast<Controller *>(handle);
	auto g = TransducerTestGain::Create(deviceIndex * 249 + transIndex, amp, phase);
	cnt->AppendGainSync(g);
}
void AUTDFlush(AUTDControllerHandle handle)
{
	auto *cnt = static_cast<Controller *>(handle);
	cnt->Flush();
}
int AUTDDevIdForDeviceIdx(AUTDControllerHandle handle, int devIdx)
{
	auto *cnt = static_cast<Controller *>(handle);
	return cnt->geometry()->deviceIdForDeviceIdx(devIdx);
}
int AUTDDevIdForTransIdx(AUTDControllerHandle handle, int transIdx)
{
	auto *cnt = static_cast<Controller *>(handle);
	return cnt->geometry()->deviceIdForTransIdx(transIdx);
}
float *AUTDTransPosition(AUTDControllerHandle handle, int transIdx)
{
	auto *cnt = static_cast<Controller *>(handle);
	auto pos = cnt->geometry()->position(transIdx);
	auto *array = new float[3];
	array[0] = pos[0];
	array[1] = pos[1];
	array[2] = pos[2];
	return array;
}
float *AUTDTransDirection(AUTDControllerHandle handle, int transIdx)
{
	auto *cnt = static_cast<Controller *>(handle);
	auto dir = cnt->geometry()->direction(transIdx);
	auto *array = new float[3];
	array[0] = dir[0];
	array[1] = dir[1];
	array[2] = dir[2];
	return array;
}
float *GetAngleZYZ(float *rotationMatrix)
{
	Matrix3f rot;
	for (int i = 0; i < 9; i++)
		rot(i / 3, i % 3) = rotationMatrix[i];
	auto euler = rot.eulerAngles(2, 1, 2);
	auto *angleZYZ = new float[3];
	angleZYZ[0] = euler[0];
	angleZYZ[1] = euler[1];
	angleZYZ[2] = euler[2];
	return angleZYZ;
}

#pragma region deprecated
void AUTDAppendLateralGain(AUTDControllerHandle handle, AUTDGainPtr gain)
{
	auto *cnt = static_cast<Controller *>(handle);
	auto *g = static_cast<Gain *>(gain);
	cnt->AppendLateralGain(g);
}
void AUTDStartLateralModulation(AUTDControllerHandle handle, float freq)
{
	auto *cnt = static_cast<Controller *>(handle);
	cnt->StartLateralModulation(freq);
}
void AUTDFinishLateralModulation(AUTDControllerHandle handle)
{
	auto *cnt = static_cast<Controller *>(handle);
	cnt->FinishLateralModulation();
}
void AUTDResetLateralGain(AUTDControllerHandle handle)
{
	auto *cnt = static_cast<Controller *>(handle);
	cnt->ResetLateralGain();
}
#pragma endregion

#pragma endregion

#pragma region Debug
#ifdef UNITY_DEBUG
DebugLogFunc _debugLogFunc = nullptr;

void DebugLog(const char *msg)
{
	if (_debugLogFunc != nullptr)
		_debugLogFunc(msg);
}

void SetDebugLog(DebugLogFunc func)
{
	_debugLogFunc = func;
}

void DebugLogTest()
{
	DebugLog("Debug Log Test");
}
#endif

#pragma endregion