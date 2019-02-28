/*
*
*  Created by Shun Suzuki on 02/07/2018.
*  Copyright © 2018 Hapis Lab. All rights reserved.
*
*/

#ifndef autd3capih_
#define autd3capih_

#ifdef _DEBUG
#define UNITY_DEBUG
#endif

extern "C" {
	using AUTDControllerHandle = void*;
	using AUTDGainPtr = void*;
	using AUTDModulationPtr = void*;

#ifdef UNITY_DEBUG
	using DebugLogFunc = void(*)(const char*);
#endif

#pragma region Controller
	__declspec(dllexport) void AUTDCreateController(AUTDControllerHandle *out);
	__declspec(dllexport) int AUTDOpenController(AUTDControllerHandle handle, const char* location);
	__declspec(dllexport) int AUTDAddDevice(AUTDControllerHandle handle, float x, float y, float z, float rz1, float ry, float rz2, int groupId);
	__declspec(dllexport) int AUTDAddDeviceQuaternion(AUTDControllerHandle handle, float x, float y, float z, float qua_w, float qua_x, float qua_y, float qua_z, int groupId);
	__declspec(dllexport) void AUTDDelDevice(AUTDControllerHandle handle, int devId);
	__declspec(dllexport) void AUTDCloseController(AUTDControllerHandle handle);
	__declspec(dllexport) void AUTDFreeController(AUTDControllerHandle handle);
	__declspec(dllexport) void AUTDSetSilentMode(AUTDControllerHandle handle, bool mode);
#pragma endregion

#pragma region Property
	__declspec(dllexport) bool AUTDIsOpen(AUTDControllerHandle handle);
	__declspec(dllexport) bool AUTDIsSilentMode(AUTDControllerHandle handle);
	__declspec(dllexport) int AUTDNumDevices(AUTDControllerHandle handle);
	__declspec(dllexport) int AUTDNumTransducers(AUTDControllerHandle handle);
	__declspec(dllexport) float AUTDFreqency(AUTDControllerHandle handle);
	__declspec(dllexport) size_t AUTDRemainingInBuffer(AUTDControllerHandle handle);
#pragma endregion

#pragma region Gain
	__declspec(dllexport) void AUTDFocalPointGain(AUTDGainPtr *gain, float x, float y, float z, uint8_t amp);
	__declspec(dllexport) void AUTDGroupedGain(AUTDGainPtr *gain, int* groupIDs, AUTDGainPtr* gains, int size);
	__declspec(dllexport) void AUTDBesselBeamGain(AUTDGainPtr *gain, float x, float y, float z, float n_x, float n_y, float n_z, float theta_z);
	__declspec(dllexport) void AUTDPlaneWaveGain(AUTDGainPtr *gain, float n_x, float n_y, float n_z);
	__declspec(dllexport) void AUTDMatlabGain(AUTDGainPtr *gain, const char * filename, const char * varname);
	__declspec(dllexport) void AUTDCustomGain(AUTDGainPtr *gain, uint16_t* data, int dataLength);
	__declspec(dllexport) void AUTDHoloGain(AUTDGainPtr *gain, float* points, float* amps, int size);
	__declspec(dllexport) void AUTDNullGain(AUTDGainPtr *gain);
	__declspec(dllexport) void AUTDDeleteGain(AUTDGainPtr gain);
	__declspec(dllexport) void AUTDFixGain(AUTDGainPtr gain);
#pragma endregion

#pragma region Modulation
	__declspec(dllexport) void AUTDModulation(AUTDModulationPtr *mod, uint8_t amp);
	__declspec(dllexport) void AUTDRawPCMModulation(AUTDModulationPtr *mod, const char * filename, float sampFreq);
	__declspec(dllexport) void AUTDSawModulation(AUTDModulationPtr *mod, float freq);
	__declspec(dllexport) void AUTDSineModulation(AUTDModulationPtr *mod, float freq, float amp, float offset);
	__declspec(dllexport) void AUTDDeleteModulation(AUTDModulationPtr mod);
#pragma endregion

#pragma region LowLevelInterface
	__declspec(dllexport) void AUTDAppendGain(AUTDControllerHandle handle, AUTDGainPtr gain);
	__declspec(dllexport) void AUTDAppendGainSync(AUTDControllerHandle handle, AUTDGainPtr gain);
	__declspec(dllexport) void AUTDAppendModulation(AUTDControllerHandle handle, AUTDModulationPtr mod);
	__declspec(dllexport) void AUTDAppendModulationSync(AUTDControllerHandle handle, AUTDModulationPtr mod);
	__declspec(dllexport) void AUTDAppendLateralGain(AUTDControllerHandle handle, AUTDGainPtr gain);
	__declspec(dllexport) void AUTDStartLateralModulation(AUTDControllerHandle handle, float freq);
	__declspec(dllexport) void AUTDFinishLateralModulation(AUTDControllerHandle handle);
	__declspec(dllexport) void AUTDResetLateralGain(AUTDControllerHandle handle);
	__declspec(dllexport) void AUTDSetGain(AUTDControllerHandle handle, int deviceIndex, int transIndex, int amp, int phase);
	__declspec(dllexport) void AUTDFlush(AUTDControllerHandle handle);
	__declspec(dllexport) int AUTDDevIdForDeviceIdx(AUTDControllerHandle handle, int devIdx);
	__declspec(dllexport) int AUTDDevIdForTransIdx(AUTDControllerHandle handle, int transIdx);
	__declspec(dllexport) float* AUTDTransPosition(AUTDControllerHandle handle, int transIdx);
	__declspec(dllexport) float* AUTDTransDirection(AUTDControllerHandle handle, int transIdx);
	__declspec(dllexport) double* GetAngleZYZ(double* rotationMatrix);
#pragma endregion

#pragma region Debug
#ifdef UNITY_DEBUG
	__declspec(dllexport) void DebugLog(const char* msg);
	__declspec(dllexport) void SetDebugLog(DebugLogFunc func);
	__declspec(dllexport) void DebugLogTest();
#endif
#pragma endregion
}

#endif