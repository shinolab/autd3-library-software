# File: NativeMethods.jl
# Project: src
# Created Date: 11/02/2020
# Author: Shun Suzuki
# -----
# Last Modified: 11/02/2020
# Modified By: Shun Suzuki (suzuki@hapis.k.u-tokyo.ac.jp)
# -----
# Copyright (c) 2020 Hapis Lab. All rights reserved.
# 

const dll_name = "./dll/autd3capi.dll"

# Controller
autd_create_controller(handle_ptr) = ccall((:AUTDCreateController, dll_name), Cvoid, (Ref{Ptr{Cvoid}},), handle_ptr)
autd_open(handle_ptr, linktype, location) = ccall((:AUTDOpenController, dll_name), Int32, (Ptr{Cvoid}, Int32, Cstring), handle_ptr, linktype, location)
autd_close(handle_ptr) = ccall((:AUTDCloseController, dll_name), Cvoid, (Ptr{Cvoid},), handle_ptr)
autd_free(handle_ptr) = ccall((:AUTDFreeController, dll_name), Cvoid, (Ptr{Cvoid},), handle_ptr)

autd_get_adapter_pointer(handle_ptr) = ccall((:AUTDGetAdapterPointer, dll_name), Int32, (Ref{Ptr{Cvoid}},), handle_ptr)
autd_get_adapter(handle, index, decs_p, name_p) = ccall((:AUTDGetAdapter, dll_name), Cvoid, (Ptr{Cvoid}, Int32, Ref{UInt8}, Ref{UInt8}), handle, index, decs_p, name_p)
autd_free_adapter_pointer(handle) = ccall((:AUTDFreeAdapterPointer, dll_name), Cvoid, (Ptr{Cvoid},), handle)

autd_add_device(handle_ptr, x, y, z, rz1, ry, rz2, id) = ccall((:AUTDAddDevice, dll_name), Int32,
        (Ptr{Cvoid}, Float32, Float32, Float32, Float32, Float32, Float32, Int32,),
        handle_ptr, x, y, z, rz1, ry, rz2, id)

autd_calibrate_modulation(handle_ptr) = ccall((:AUTDCalibrateModulation, dll_name), Cvoid, (Ptr{Cvoid},), handle_ptr)

# Gain
autd_focal_point_gain(gain_ptr, x, y, z, amp) = ccall((:AUTDFocalPointGain, dll_name), Cvoid, 
        (Ref{Ptr{Cvoid}}, Float32, Float32, Float32, UInt8), 
        gain_ptr, x, y, z, amp)

autd_delete_gain(gain_ptr) = ccall((:AUTDDeleteGain, dll_name), Cvoid,  (Ptr{Cvoid},), gain_ptr)

# Modulation
autd_modulation(mod_ptr, amp) = ccall((:AUTDModulation, dll_name), Cvoid, 
        (Ref{Ptr{Cvoid}}, UInt8), 
        mod_ptr, amp)

autd_sine_modulation(mod_ptr, freq, amp, offset) = ccall((:AUTDSineModulation, dll_name), Cvoid, 
        (Ref{Ptr{Cvoid}}, Int32, Float32, Float32), 
        mod_ptr, freq, amp, offset)

autd_delete_modulation(gain_ptr) = ccall((:AUTDDeleteModulation, dll_name), Cvoid,  (Ptr{Cvoid},), gain_ptr)

# Low Level Interface
autd_append_gain(handle_ptr, gain_ptr) = ccall((:AUTDAppendGain, dll_name), Cvoid, (Ptr{Cvoid}, Ptr{Cvoid}), handle_ptr, gain_ptr)
autd_append_gain_sync(handle_ptr, gain_ptr) = ccall((:AUTDAppendGainSync, dll_name), Cvoid, (Ptr{Cvoid}, Ptr{Cvoid}), handle_ptr, gain_ptr)
autd_append_modulation(handle_ptr, mod_ptr) = ccall((:AUTDAppendModulation, dll_name), Cvoid, (Ptr{Cvoid}, Ptr{Cvoid}), handle_ptr, mod_ptr)
autd_append_modulation_sync(handle_ptr, mod_ptr) = ccall((:AUTDAppendModulationSync, dll_name), Cvoid, (Ptr{Cvoid}, Ptr{Cvoid}), handle_ptr, mod_ptr)

autd_append_stm_gain(handle_ptr, gain_ptr) = ccall((:AUTDAppendSTMGain, dll_name), Cvoid, (Ptr{Cvoid}, Ptr{Cvoid}), handle_ptr, gain_ptr)
autd_start_stm(handle_ptr, freq) = ccall((:AUTDStartSTModulation, dll_name), Cvoid, (Ptr{Cvoid}, Float32), handle_ptr, freq)
autd_stop_stm(handle_ptr) = ccall((:AUTDStopSTModulation, dll_name), Cvoid, (Ptr{Cvoid},), handle_ptr)
autd_finish_stm(handle_ptr) = ccall((:AUTDFinishSTModulation, dll_name), Cvoid, (Ptr{Cvoid},), handle_ptr)
