'''
File: nativemethods.py
Project: pyautd
Created Date: 11/02/2020
Author: Shun Suzuki
-----
Last Modified: 11/02/2020
Modified By: Shun Suzuki (suzuki@hapis.k.u-tokyo.ac.jp)
-----
Copyright (c) 2020 Hapis Lab. All rights reserved.

'''

import ctypes
from ctypes import c_void_p, c_bool, c_int, create_string_buffer, byref, POINTER, c_float, c_long, c_char, c_char_p, c_ubyte


def init_autd3(dlllocation="./autd3capi.dll"):
    global autddll
    autddll = ctypes.CDLL(dlllocation)

    __init_controller()
    __init_property()
    __init_gain()
    __init_modulation()
    __init_low_level_interface()


def __init_controller():
    autddll.AUTDCreateController.argtypes = [POINTER(c_void_p)]
    autddll.AUTDCreateController.restypes = [None]

    autddll.AUTDOpenController.argtypes = [c_void_p, c_int, c_char_p]
    autddll.AUTDOpenController.restypes = [c_int]

    autddll.AUTDGetAdapterPointer.argtypes = [POINTER(c_void_p)]
    autddll.AUTDGetAdapterPointer.restypes = [c_int]

    autddll.AUTDGetAdapter.argtypes = [
        c_void_p, c_int, c_char_p, c_char_p]
    autddll.AUTDGetAdapter.restypes = [None]

    autddll.AUTDFreeAdapterPointer.argtypes = [c_void_p]
    autddll.AUTDFreeAdapterPointer.restypes = [None]

    autddll.AUTDAddDevice.argtypes = [
        c_void_p, c_float, c_float, c_float, c_float, c_float, c_float, c_int]
    autddll.AUTDAddDevice.restypes = [c_int]

    autddll.AUTDAddDeviceQuaternion.argtypes = [
        c_void_p, c_float, c_float, c_float, c_float, c_float, c_float,  c_float, c_int]
    autddll.AUTDAddDeviceQuaternion.restypes = [c_int]

    autddll.AUTDDelDevice.argtypes = [c_void_p, c_int]
    autddll.AUTDDelDevice.restypes = [None]

    autddll.AUTDCloseController.argtypes = [c_void_p]
    autddll.AUTDCloseController.restypes = [None]

    autddll.AUTDFreeController.argtypes = [c_void_p]
    autddll.AUTDFreeController.restypes = [None]

    autddll.AUTDSetSilentMode.argtypes = [c_void_p, c_bool]
    autddll.AUTDSetSilentMode.restypes = [None]

    autddll.AUTDCalibrateModulation.argtypes = [c_void_p]
    autddll.AUTDCalibrateModulation.restypes = [None]


def __init_property():
    autddll.AUTDIsOpen.argtypes = [c_void_p]
    autddll.AUTDIsOpen.restypes = [c_bool]

    autddll.AUTDIsSilentMode.argtypes = [c_void_p]
    autddll.AUTDIsSilentMode.restypes = [c_bool]

    autddll.AUTDNumDevices.argtypes = [c_void_p]
    autddll.AUTDNumDevices.restypes = [c_int]

    autddll.AUTDNumTransducers.argtypes = [c_void_p]
    autddll.AUTDNumTransducers.restypes = [c_int]

    autddll.AUTDFrequency.argtypes = [c_void_p]
    autddll.AUTDFrequency.restypes = [c_float]

    autddll.AUTDRemainingInBuffer.argtypes = [c_void_p]
    autddll.AUTDRemainingInBuffer.restypes = [c_long]


def __init_gain():
    autddll.AUTDFocalPointGain.argtypes = [
        POINTER(c_void_p), c_float, c_float, c_float, c_ubyte]
    autddll.AUTDFocalPointGain.restypes = [None]

    autddll.AUTDGroupedGain.argtypes = [
        POINTER(c_void_p), POINTER(c_int), POINTER(c_void_p), c_int]
    autddll.AUTDGroupedGain.restypes = [None]

    autddll.AUTDBesselBeamGain.argtypes = [
        POINTER(c_void_p), c_float, c_float, c_float, c_float, c_float, c_float, c_float]
    autddll.AUTDBesselBeamGain.restypes = [None]

    autddll.AUTDPlaneWaveGain.argtypes = [
        POINTER(c_void_p), c_float, c_float, c_float]
    autddll.AUTDPlaneWaveGain.restypes = [None]

    autddll.AUTDMatlabGain.argtypes = [
        POINTER(c_void_p), c_char_p, c_char_p]
    autddll.AUTDMatlabGain.restypes = [None]

    autddll.AUTDCustomGain.argtypes = [
        POINTER(c_void_p), POINTER(c_ubyte), c_int]
    autddll.AUTDCustomGain.restypes = [None]

    autddll.AUTDHoloGain.argtypes = [
        POINTER(c_void_p), POINTER(c_float), POINTER(c_float), c_int]
    autddll.AUTDHoloGain.restypes = [None]

    autddll.AUTDTransducerTestGain.argtypes = [
        POINTER(c_void_p), c_int, c_int, c_int]
    autddll.AUTDTransducerTestGain.restypes = [None]

    autddll.AUTDNullGain.argtypes = [
        POINTER(c_void_p)]
    autddll.AUTDNullGain.restypes = [None]

    autddll.AUTDDeleteGain.argtypes = [c_void_p]
    autddll.AUTDDeleteGain.restypes = [None]


def __init_modulation():
    autddll.AUTDModulation.argtypes = [
        POINTER(c_void_p), c_ubyte]
    autddll.AUTDModulation.restypes = [None]

    autddll.AUTDRawPCMModulation.argtypes = [
        POINTER(c_void_p), c_char_p, c_float]
    autddll.AUTDRawPCMModulation.restypes = [None]

    autddll.AUTDRawPCMModulation.argtypes = [
        POINTER(c_void_p), c_int]
    autddll.AUTDRawPCMModulation.restypes = [None]

    autddll.AUTDSineModulation.argtypes = [
        POINTER(c_void_p), c_int, c_float, c_float]
    autddll.AUTDSineModulation.restypes = [None]

    autddll.AUTDDeleteModulation.argtypes = [c_void_p]
    autddll.AUTDDeleteModulation.restypes = [None]


def __init_low_level_interface():
    autddll.AUTDAppendGain.argtypes = [
        c_void_p, c_void_p]
    autddll.AUTDAppendGain.restypes = [None]

    autddll.AUTDAppendGainSync.argtypes = [
        c_void_p, c_void_p]
    autddll.AUTDAppendGainSync.restypes = [None]

    autddll.AUTDAppendModulation.argtypes = [
        c_void_p, c_void_p]
    autddll.AUTDAppendModulation.restypes = [None]

    autddll.AUTDAppendModulationSync.argtypes = [
        c_void_p, c_void_p]
    autddll.AUTDAppendModulationSync.restypes = [None]

    autddll.AUTDAppendSTMGain.argtypes = [
        c_void_p, c_void_p]
    autddll.AUTDAppendSTMGain.restypes = [None]

    autddll.AUTDStartSTModulation.argtypes = [
        c_void_p, c_float]
    autddll.AUTDStartSTModulation.restypes = [None]

    autddll.AUTDStopSTModulation.argtypes = [
        c_void_p]
    autddll.AUTDStopSTModulation.restypes = [None]

    autddll.AUTDFinishSTModulation.argtypes = [
        c_void_p]
    autddll.AUTDFinishSTModulation.restypes = [None]
