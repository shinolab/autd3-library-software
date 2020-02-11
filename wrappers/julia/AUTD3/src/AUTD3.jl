# File: AUTD3.jl
# Project: src
# Created Date: 11/02/2020
# Author: Shun Suzuki
# -----
# Last Modified: 11/02/2020
# Modified By: Shun Suzuki (suzuki@hapis.k.u-tokyo.ac.jp)
# -----
# Copyright (c) 2020 Hapis Lab. All rights reserved.
# 

module AUTD3

export Gain, Modulation, AUTD
export open_autd, dispose, add_device, calibrate_modulation
export enumerate_adapters
export focal_point_gain
export sine_modulation, modulation
export append_gain_sync, append_modulation_sync
export append_stm_gain, start_stm, stop_stm, finish_stm

include("NativeMethods.jl")

macro exported_enum(name, args...)
    esc(quote
        @enum($name, $(args...))
        export $name
        $([:(export $arg) for arg in args]...)
    end)
end

@exported_enum LinkType EtherCAT TwinCAT SOEM

mutable struct Gain
    _gain_ptr::Ptr{Cvoid}
    _disposed::Bool
    function Gain(gain_ptr::Ptr{Cvoid})
        gain = new(gain_ptr, false)
        finalizer(gain->dispose(gain), gain)
        gain
    end
end

mutable struct Modulation
    _mod_ptr::Ptr{Cvoid}
    _disposed::Bool
    function Modulation(mod_ptr::Ptr{Cvoid})
        modulation = new(mod_ptr, false)
        finalizer(modulation->dispose(modulation), modulation)
        modulation
    end
end

mutable struct AUTD
    _handle::Ptr{Cvoid}
    _disposed::Bool
    function AUTD()
        chandle = Ref(Ptr{Cvoid}(0))
        autd_create_controller(chandle)
        autd = new(chandle[], false)
        finalizer(autd->dispose(autd), autd)
        autd
    end
end

function open_autd(autd::AUTD, linktype::LinkType, location = "")
    autd_open(autd._handle, Int32(linktype), location)
end

function dispose(self::AUTD)
    autd_close(self._handle)
    autd_free(self._handle)
    self._disposed = true
end

function dispose(self::Gain)
    autd_delete_gain(self._gain_ptr)
    self._disposed = true
end

function dispose(self::Modulation)
    autd_delete_modulation(self._mod_ptr)
    self._disposed = true
end

function enumerate_adapters()
    res = []
    phandle = Ref(Ptr{Cvoid}(0))
    size = autd_get_adapter_pointer(phandle)
    handle::Ptr{Cvoid} = phandle[]

    for i in 0:size - 1
        sb_desc = zeros(UInt8, 128)
        sb_name = zeros(UInt8, 128)
        autd_get_adapter(handle, i, sb_desc,  sb_name)
        push!(res, [String(strip(String(sb_desc), '\0')), String(strip(String(sb_name), '\0'))])
    end

    autd_free_adapter_pointer(handle)
    res
end

function add_device(autd::AUTD, pos::Tuple{Float32,Float32,Float32} = (0.f0, 0.f0, 0.f0), rot::Tuple{Float32,Float32,Float32} = (0.f0, 0.f0, 0.f0))
    x, y, z = pos
    az1, ay, az2 = rot
    autd_add_device(autd._handle, x, y, z, az1, ay, az2, 1)
end

function calibrate_modulation(autd::AUTD)
    autd_calibrate_modulation(autd._handle)
end

function focal_point_gain(position::Tuple{Float32,Float32,Float32}; amp = UInt8(255))
    x, y, z = position
    chandle = Ref(Ptr{Cvoid}(0))
    autd_focal_point_gain(chandle, x, y, z, UInt8(amp))
    Gain(chandle[])
end

function modulation(amp = 0xFF)
    chandle = Ref(Ptr{Cvoid}(0))
    autd_modulation(chandle, UInt8(amp))
    Modulation(chandle[])
end

function sine_modulation(freq, amp = 1.0f0, offset = 0.5f0)
    chandle = Ref(Ptr{Cvoid}(0))
    autd_sine_modulation(chandle, Int32(freq), amp, offset)
    Modulation(chandle[])
end

function append_gain(autd::AUTD, gain::Gain)
    autd_append_gain(autd._handle, gain._gain_ptr);
end

function append_gain_sync(autd::AUTD, gain::Gain)
    autd_append_gain_sync(autd._handle, gain._gain_ptr);
end

function append_modulation(autd::AUTD, mod::Modulation)
    autd_append_modulation(autd._handle, mod._mod_ptr);
end

function append_modulation_sync(autd::AUTD, mod::Modulation)
    autd_append_modulation_sync(autd._handle, mod._mod_ptr);
end

function append_stm_gain(autd::AUTD, gain::Gain)
    autd_append_stm_gain(autd._handle, gain._gain_ptr);
end

function start_stm(autd::AUTD, freq::Float32)
    autd_start_stm(autd._handle, freq);
end

function stop_stm(autd::AUTD)
    autd_stop_stm(autd._handle);
end

function finish_stm(autd::AUTD)
    autd_finish_stm(autd._handle);
end

end
