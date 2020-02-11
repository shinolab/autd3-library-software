'''
File: autd.py
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
from ctypes import c_void_p, byref, create_string_buffer, c_char, c_char_p
import sys
from enum import IntEnum

from . import nativemethods


class LinkType(IntEnum):
    ETHERCAT = 0
    TwinCAT = 1
    SOEM = 2


class Gain:
    def __init__(self):
        self.gain_ptr = c_void_p()

    def __del__(self):
        nativemethods.autddll.AUTDDeleteGain(self.gain_ptr)


class Modulation:
    def __init__(self):
        self.modulation_ptr = c_void_p()

    def __del__(self):
        nativemethods.autddll.AUTDDeleteModulation(self.modulation_ptr)


class AUTD:
    def __init__(self):
        self.autd = c_void_p()
        nativemethods.autddll.AUTDCreateController(byref(self.autd))

        self.__disposed = False

    def __del__(self):
        self.dispose()

    def open(self, linktype=LinkType.SOEM, location=""):
        nativemethods.autddll.AUTDOpenController(
            self.autd, int(linktype), location.encode('utf-8'))

    @staticmethod
    def enumerate_adapters():
        res = []
        handle = c_void_p()
        size = nativemethods.autddll.AUTDGetAdapterPointer(byref(handle))

        for i in range(size):
            sb_desc = ctypes.create_string_buffer(128)
            sb_name = ctypes.create_string_buffer(128)
            nativemethods.autddll.AUTDGetAdapter(
                handle, i, sb_desc, sb_name)
            res.append([sb_name.value.decode('utf-8'),
                        sb_desc.value.decode('utf-8')])

        nativemethods.autddll.AUTDFreeAdapterPointer(handle)

        return res

    def close(self):
        nativemethods.autddll.AUTDCloseController(self.autd)

    def free(self):
        if not self.__disposed:
            nativemethods.autddll.AUTDFreeController(self.autd)
            self.__disposed = True

    def dispose(self):
        self.close()
        self.free()

    def set_silent(self, silent: bool):
        nativemethods.autddll.AUTDSetSilentMode(self.autd, silent)

    def calibrate_modulation(self, silent: bool):
        nativemethods.autddll.AUTDCalibrateModulation(self.autd)

    def add_device(self, pos=[0., 0., 0.], rot=[0., 0., 0.], id=0):
        nativemethods.autddll.AUTDAddDevice(
            self.autd, pos[0], pos[1], pos[2], rot[0], rot[1], rot[2], id)

    @staticmethod
    def focal_point_gain(x, y, z, amp=255):
        gain = Gain()
        nativemethods.autddll.AUTDFocalPointGain(
            byref(gain.gain_ptr), x, y, z, amp)
        return gain

    @staticmethod
    def modulation(amp=255):
        mod = Modulation()
        nativemethods.autddll.AUTDModulation(
            byref(mod.modulation_ptr), amp)
        return mod

    @staticmethod
    def sine_modulation(freq, amp=1.0, offset=0.5):
        mod = Modulation()
        nativemethods.autddll.AUTDSineModulation(
            byref(mod.modulation_ptr), freq, amp, offset)
        return mod

    def append_modulation(self, mod: Modulation):
        nativemethods.autddll.AUTDAppendModulation(
            self.autd, mod.modulation_ptr)

    def append_modulation_sync(self, mod: Modulation):
        nativemethods.autddll.AUTDAppendModulationSync(
            self.autd, mod.modulation_ptr)

    def append_gain(self, gain: Gain):
        nativemethods.autddll.AUTDAppendGain(self.autd, gain.gain_ptr)

    def append_gain_sync(self, gain: Gain):
        nativemethods.autddll.AUTDAppendGainSync(self.autd, gain.gain_ptr)

    def append_stm_gain(self, gain: Gain):
        nativemethods.autddll.AUTDAppendSTMGain(self.autd, gain.gain_ptr)

    def start_stm(self, freq):
        nativemethods.autddll.AUTDStartSTModulation(self.autd, freq)

    def stop_stm(self):
        nativemethods.autddll.AUTDStopSTModulation(self.autd)

    def finish_stm(self):
        nativemethods.autddll.AUTDFinishSTModulation(self.autd)
