# API Reference

The API for the C language is defined under [client/capi](https://github.com/shinolab/autd3-library-software/tree/master/client/capi).
The reference of this API is given below. 
For actual usage, see [C#](https://github.com/shinolab/autd3sharp)/[python](https://github.com/shinolab/pyautd)/[Julia](https://github.com/shinolab/AUTD3.jl) wrapper libraries.

> Note: The calling conventions are not explicitly stated. The x86 conventions are probably `cdecl`, but we haven't checked it yet, and it may cause errors when used from x86.

##  AUTDCreateController (autd3capi)

Create `Controller`.

You need to release the `Controller` you created with `AUTDFreeController` at the end.

| Argument name / return       | type             | in/out | description                                                   |
|------------------------------|------------------|--------|--------------------------------------------------------------|
| out                          | void**    | out    | pointer to ControllerPtr                                     |
| return                       | void      | -      | nothing                                                      |

##  AUTDOpenController (autd3capi)

Open `Controller`. 
The `handle` is the controller created by `AUTDCreateController`.

This function returns false if fails.
If it returns false, you can get the error message with `AUTDGetLastError`.

| Argument name / return       | type             | in/out | description                                                   |
|------------------------------|------------------|--------|--------------------------------------------------------------|
| handle                       | void*      | in     | ControllerPtr                                                |
| link                         | void*      | in     | LinkPtr                                                      |
| return                       | bool       | -      | true if success                                              |

##  AUTDAddDevice (autd3capi)

Add device to `Controller`.
The `handle` is the controller created by `AUTDCreateController`. 
(x, y, z) are the positions, and (rz1, ry, rz2) are the ZYZ euler angles.

This function returns the ID of the added device.

| Argument name / return       | type             | in/out | description                                                   |
|------------------------------|------------------|--------|--------------------------------------------------------------|
| handle                       | void*      | in     | ControllerPtr                                                |
| x                            | double     | in     | x coordinate of position in millimeter                       |
| y                            | double     | in     | y coordinate of position in millimeter                       |
| z                            | double     | in     | z coordinate of position in millimeter                       |
| rz1                          | double     | in     | first angle of ZYZ euler angle in radian                     |
| ry                           | double     | in     | second angle of ZYZ euler angle in radian                    |
| rz2                          | double     | in     | third angle of ZYZ euler angle in radian                     |
| gid                          | int32_t    | in     | group Id                                                     |
| return                       | int32_t    | -      | Device Id                                                    |

##  AUTDAddDeviceQuaternion (autd3capi)

Add device to `Controller`.
The `handle` is the controller created by `AUTDCreateController`.
(x, y, z) are the positions, and (qw, qx, qy, qz) are the quaternion.

This function returns the ID of the added device.

| Argument name / return       | type             | in/out | description                                                   |
|------------------------------|------------------|--------|--------------------------------------------------------------|
| handle                       | void*      | in     | ControllerPtr                                                |
| x                            | double     | in     | x coordinate of position in millimeter                       |
| y                            | double     | in     | y coordinate of position in millimeter                       |
| z                            | double     | in     | z coordinate of position in millimeter                       |
| qw                           | double     | in     | quaternion of rotation                                       |
| qx                           | double     | in     | quaternion of rotation                                       |
| qy                           | double     | in     | quaternion of rotation                                       |
| qz                           | double     | in     | quaternion of rotation                                       |
| gid                          | int32_t    | in     | group Id                                                     |
| return                       | int32_t    | -      | Device Id                                                    |

##  AUTDDeleteDevice (autd3capi)

Deletes a device at the specified index from the Controller.
The `handle` is the controller created by `AUTDCreateController`.

This function returns the ID of the deleted device.

| Argument name / return       | type             | in/out | description                                                   |
|------------------------------|------------------|--------|--------------------------------------------------------------|
| handle                       | void*      | in     | ControllerPtr                                                |
| idx                          | int32_t    | in     | Device index                                                 |
| return                       | int32_t    | -      | Deleted device Id                                            |

##  AUTDClearDevices (autd3capi)

Deletes all device from the Controller.
The `handle` is the controller created by `AUTDCreateController`.

| Argument name / return       | type             | in/out | description                                                   |
|------------------------------|------------------|--------|--------------------------------------------------------------|
| handle                       | void*      | in     | ControllerPtr                                                |
| return                       | void       | -      | nothing                                                      |

##  AUTDCloseController (autd3capi)

Close Controller.
The `handle` is the controller created by `AUTDCreateController`.

This function returns a value less than 0 if an error occurred.
If an error occurs, you can get the error message with `AUTDGetLastError`.
Also, if the check ack flag is on and the return value is greater than 0, it guarantees that the data has been processed by the device.

| Argument name / return       | type             | in/out | description                                                                              |
|------------------------------|------------------|--------|-----------------------------------------------------------------------------------------|
| handle                       | void*      | in     | ControllerPtr                                                                           |
| return                       | int32_t    | -      | if $>0$, it guarantees devices have processed data. if $<0$, error ocurred.             |

##  AUTDClear (autd3capi)

Clear Controller.
The `handle` is the controller created by `AUTDCreateController`.

This function returns a value less than 0 if an error occurred.
If an error occurs, you can get the error message with `AUTDGetLastError`.
Also, if the check ack flag is on and the return value is greater than 0, it guarantees that the data has been processed by the device.

| Argument name / return       | type             | in/out | description                                                                              |
|------------------------------|------------------|--------|-----------------------------------------------------------------------------------------|
| handle                       | void*      | in     | ControllerPtr                                                                           |
| return                       | int32_t    | -      | if $>0$, it guarantees devices have processed data. if $<0$, error ocurred.             |

##  AUTDFreeController (autd3capi)

Delete Controller.
The `handle` is the controller created by `AUTDCreateController`.
Never use `handle` after calling this function. 

| Argument name / return       | type             | in/out | description                                                                              |
|------------------------------|------------------|--------|-----------------------------------------------------------------------------------------|
| handle                       | void*      | in     | ControllerPtr                                                                           |
| return                       | void       | -      | nothing                                                                                 |

##  AUTDIsOpen (autd3capi)

Returns whether Controller is open or not.
The `handle` is the controller created by `AUTDCreateController`.

| Argument name / return       | type             | in/out | description                                                                              |
|------------------------------|------------------|--------|-----------------------------------------------------------------------------------------|
| handle                       | void*      | in     | ControllerPtr                                                                           |
| return                       | bool       | -      | true if controller is open                                                              |

##  AUTDGetOutputEnable (autd3capi)

Returns Output enable flag.
The `handle` is the controller created by `AUTDCreateController`.

| Argument name / return       | type             | in/out | description                                                                              |
|------------------------------|------------------|--------|-----------------------------------------------------------------------------------------|
| handle                       | void*      | in     | ControllerPtr                                                                           |
| return                       | bool       | -      | output enable flag                                                                      |

##  AUTDGetSilentMode (autd3capi)

Returns silent mode flag.
The `handle` is the controller created by `AUTDCreateController`.

| Argument name / return       | type             | in/out | description                                                                              |
|------------------------------|------------------|--------|-----------------------------------------------------------------------------------------|
| handle                       | void*      | in     | ControllerPtr                                                                           |
| return                       | bool       | -      | silent mode flag                                                                        |

##  AUTDGetForceFan (autd3capi)

Returns Force fan flag.
 The `handle` is the controller created by `AUTDCreateController`.

| Argument name / return       | type             | in/out | description                                                                              |
|------------------------------|------------------|--------|-----------------------------------------------------------------------------------------|
| handle                       | void*      | in     | ControllerPtr                                                                           |
| return                       | bool       | -      | Force fan flag                                                                          |

##  AUTDGetReadsFPGAInfo (autd3capi)

Returns reads FPGA Info flag.
The `handle` is the controller created by `AUTDCreateController`.

| Argument name / return       | type             | in/out | description                                                                              |
|------------------------------|------------------|--------|-----------------------------------------------------------------------------------------|
| handle                       | void*      | in     | ControllerPtr                                                                           |
| return                       | bool       | -      | reads FPGA Info flag                                                                    |

##  AUTDGetOutputBalance (autd3capi)

Returns Output balance flag.
The `handle` is the controller created by `AUTDCreateController`.

| Argument name / return       | type             | in/out | description                                                                              |
|------------------------------|------------------|--------|-----------------------------------------------------------------------------------------|
| handle                       | void*      | in     | ControllerPtr                                                                           |
| return                       | bool       | -      | Output balance flag                                                                     |

##  AUTDGetCheckAck (autd3capi)

Returns check ack flag.
The `handle` is the controller created by `AUTDCreateController`.

| Argument name / return       | type             | in/out | description                                                                              |
|------------------------------|------------------|--------|-----------------------------------------------------------------------------------------|
| handle                       | void*      | in     | ControllerPtr                                                                           |
| return                       | bool       | -      | Check ack flag                                                                          |

##  AUTDSetOutputEnable (autd3capi)

Set Output enable flag.
The `handle` is the controller created by `AUTDCreateController`.

This flag will be updated in the device after calling one of the send functions.

| Argument name / return       | type             | in/out | description                                                                              |
|------------------------------|------------------|--------|-----------------------------------------------------------------------------------------|
| handle                       | void*      | in     | ControllerPtr                                                                           |
| enable                       | bool       | in     | output enable flag                                                                      |
| return                       | void       | -      | nothing                                                                                 |

##  AUTDSetSilentMode (autd3capi)

Set silent mode flag.
The `handle` is the controller created by `AUTDCreateController`.

This flag will be updated in the device after calling one of the send functions.

| Argument name / return       | type             | in/out | description                                                                              |
|------------------------------|------------------|--------|-----------------------------------------------------------------------------------------|
| handle                       | void*      | in     | ControllerPtr                                                                           |
| mode                         | bool       | in     | silent mode flag                                                                        |
| return                       | void       | -      | nothing                                                                                 |

##  AUTDSetReadsFPGAInfo (autd3capi)

Set reads FPGA Info flag.
The `handle` is the controller created by `AUTDCreateController`.

This flag will be updated in the device after calling one of the send functions.

| Argument name / return       | type             | in/out | description                                                                              |
|------------------------------|------------------|--------|-----------------------------------------------------------------------------------------|
| handle                       | void*      | in     | ControllerPtr                                                                           |
| reads\_fpga\_info            | bool       | in     | read FPGA info flag                                                                     |
| return                       | void       | -      | nothing                                                                                 |

##  AUTDSetForceFan (autd3capi)

Set Force fan flag.
The `handle` is the controller created by `AUTDCreateController`.

This flag will be updated in the device after calling one of the send functions.

| Argument name / return       | type             | in/out | description                                                                              |
|------------------------------|------------------|--------|-----------------------------------------------------------------------------------------|
| handle                       | void*      | in     | ControllerPtr                                                                           |
| force                        | bool       | in     | force fan flag                                                                          |
| return                       | void       | -      | nothing                                                                                 |

##  AUTDSetOutputBalance (autd3capi)

Set Output balance flag.
The `handle` is the controller created by `AUTDCreateController`.

This flag will be updated in the device after calling one of the send functions.

| Argument name / return       | type             | in/out | description                                                                              |
|------------------------------|------------------|--------|-----------------------------------------------------------------------------------------|
| handle                       | void*      | in     | ControllerPtr                                                                           |
| output_balance               | bool       | in     | Output balance flag                                                                     |
| return                       | void       | -      | nothing                                                                                 |

##  AUTDSetCheckAck (autd3capi)

Set Check ack flag.
The `handle` is the controller created by `AUTDCreateController`.

| Argument name / return       | type             | in/out | description                                                                              |
|------------------------------|------------------|--------|-----------------------------------------------------------------------------------------|
| handle                       | void*      | in     | ControllerPtr                                                                           |
| check_ack                    | bool       | in     | Check ack flag                                                                          |
| return                       | void       | -      | nothing                                                                                 |

##  AUTDGetWavelength (autd3capi)

Get wavelength.
The `handle` is the controller created by `AUTDCreateController`.

| Argument name / return       | type             | in/out | description                                                                              |
|------------------------------|------------------|--------|-----------------------------------------------------------------------------------------|
| handle                       | void*      | in     | ControllerPtr                                                                           |
| return                       | double     | -      | wavelength                                                                              |

##  AUTDGetAttenuation (autd3capi)

Get attenuation coefficient.
The `handle` is the controller created by `AUTDCreateController`.

| Argument name / return       | type             | in/out | description                                                                              |
|------------------------------|------------------|--------|-----------------------------------------------------------------------------------------|
| handle                       | void*      | in     | ControllerPtr                                                                           |
| return                       | double     | -      | attenuation coefficient                                                                 |

##  AUTDSetWavelength (autd3capi)

Set wavelength.
The `handle` is the controller created by `AUTDCreateController`.

| Argument name / return       | type             | in/out | description                                                                              |
|------------------------------|------------------|--------|-----------------------------------------------------------------------------------------|
| handle                       | void*      | in     | ControllerPtr                                                                           |
| wavelength                   | double     | in     | wavelength                                                                              |
| return                       | void       | -      | nothing                                                                                 |

##  AUTDSetAttenuation (autd3capi)

Set attenuation coefficient.
The `handle` is the controller created by `AUTDCreateController`.

| Argument name / return       | type             | in/out | description                                                                              |
|------------------------------|------------------|--------|-----------------------------------------------------------------------------------------|
| handle                       | void*      | in     | ControllerPtr                                                                           |
| attenuation                  | double     | in     | attenuation coefficient                                                                 |
| return                       | void       | -      | nothing                                                                                 |

##  AUTDGetFPGAInfo (autd3capi)

Get the information of FPGAs.
The `handle` is the controller created by `AUTDCreateController`.
The memory pointed by the `out` pointer should be the same length as the connected device.
In the FPGAs information, the lowest one bit indicates whether the fan is running or not,
The other bits are all 0.

Before calling this function, you need to set the read FPGA info
flag must be set to on.

This function returns false if it fails.
If it returns false, you can get the error message with `AUTDGetLastError`.

| Argument name / return       | type             | in/out | description                                                                              |
|------------------------------|------------------|--------|-----------------------------------------------------------------------------------------|
| handle                       | void*      | in     | ControllerPtr                                                                           |
| out                          | uint8_t *  | out    | FPGA informations                                                                        |
| return                       | bool       | -      | true if success                                                                         |

##  AUTDUpdateCtrlFlags (autd3capi)

Update control flags. One of the send functions.
After setting output enable, silent mode, force
fan, and reads FPGA info flags, these changes will be actually updated in the devices.
The `handle` is the controller created by `AUTDCreateController`.

This function returns a value less than 0 if an error occurred.
If an error occurs, you can get the error message with `AUTDGetLastError`.
Also, if the check ack flag is on and the return value is greater than 0, it guarantees that the data has been processed by the device.

| Argument name / return       | type             | in/out | description                                                                              |
|------------------------------|------------------|--------|-----------------------------------------------------------------------------------------|
| handle                       | void*      | in     | ControllerPtr                                                                           |
| return                       | int32_t    | -      | if $>0$, it guarantees devices have processed data. if $<0$, error ocurred.             |

##  AUTDSetOutputDelay (autd3capi)

Set output delays. One of the send functions.
The `handle` is the controller created by `AUTDCreateController`.
The `delay` must be a pointer to data of length (number of devices) $\times 249$.

This function returns a value less than 0 if an error occurred.
If an error occurs, you can get the error message with `AUTDGetLastError`.
Also, if the check ack flag is on and the return value is greater than 0, it guarantees that the data has been processed by the device.

| Argument name / return       | type             | in/out | description                                                                              |
|------------------------------|------------------|--------|-----------------------------------------------------------------------------------------|
| handle                       | void*      | in     | ControllerPtr                                                                           |
| delay                        | uint8_t *  | in     | pointer to delay data                                                                   |
| return                       | int32_t    | -      | if $>0$, it guarantees devices have processed data. if $<0$, error ocurred.             |

##  AUTDSetDutyOffset (autd3capi)

Set duty offset. One of the send functions.
The `handle` is the controller created by `AUTDCreateController`.
The `offset` must be a pointer to data of length (number of devices) $\times 249$.

This function returns a value less than 0 if an error occurred.
If an error occurs, you can get the error message with `AUTDGetLastError`.
Also, if the check ack flag is on and the return value is greater than 0, it guarantees that the data has been processed by the device.

| Argument name / return       | type             | in/out | description                                                                              |
|------------------------------|------------------|--------|-----------------------------------------------------------------------------------------|
| handle                       | void*      | in     | ControllerPtr                                                                           |
| offset                       | uint8_t *  | in     | pointer to duty offset data                                                             |
| return                       | int32_t    | -      | if $>0$, it guarantees devices have processed data. if $<0$, error ocurred.             |

##  AUTDSetDelayOffset (autd3capi)

Set output delays and duty offsets. 
One of the send functions.
The `handle` is the controller created by `AUTDCreateController`. delay,
The `delay` and `offset` must be a pointer to data of length (number of devices) $\times 249$.

This function returns a value less than 0 if an error occurred.
If an error occurs, you can get the error message with `AUTDGetLastError`.
Also, if the check ack flag is on and the return value is greater than 0, it guarantees that the data has been processed by the device.

| Argument name / return       | type             | in/out | description                                                                              |
|------------------------------|------------------|--------|-----------------------------------------------------------------------------------------|
| handle                       | void*      | in     | ControllerPtr                                                                           |
| delay                        | uint8_t *  | in     | pointer to output delay data                                                            |
| offset                       | uint8_t *  | in     | pointer to duty offset data                                                             |
| return                       | int32_t    | -      | if $>0$, it guarantees devices have processed data. if $<0$, error ocurred.             |

##  AUTDGetLastError (autd3capi)

Get the error message that occurred last.

The error message is copied to `error` pointer. 
If the argument is nullptr, the error message is not copied. 
The function returns the length of the error message with null-terminated.

Since the length of the error message is variable, you should reserve a sufficiently large area, or, pass nullptr as `error` to get the required size and call it again.

| Argument name / return       | type             | in/out | description                                                                              |
|------------------------------|------------------|--------|-----------------------------------------------------------------------------------------|
| error                        | char*            | out    | pointer to error message                                                                |
| return                       | int32_t    | -      | length of error message including null terminator                                       |

##  AUTDNumDevices (autd3capi)

Get the number of devices connected.
The `handle` is the controller created by `AUTDCreateController`.

| Argument name / return       | type             | in/out | description                                                                              |
|------------------------------|------------------|--------|-----------------------------------------------------------------------------------------|
| handle                       | void*      | in     | ControllerPtr                                                                           |
| return                       | int32_t    | -      | number of devices                                                                       |

##  AUTDNumTransducers (autd3capi)

Get the total number of transducers.
The `handle` is the controller created by `AUTDCreateController`.

| Argument name / return       | type             | in/out | description                                                                              |
|------------------------------|------------------|--------|-----------------------------------------------------------------------------------------|
| handle                       | void*      | in     | ControllerPtr                                                                           |
| return                       | int32_t    | -      | number of transducers                                                                   |

##  AUTDDeviceIdxForTransIdx (autd3capi)

Convert global transducer index to device index. 
The `handle` is the controller created by `AUTDCreateController`.

| Argument name / return       | type             | in/out | description                                                                              |
|------------------------------|------------------|--------|-----------------------------------------------------------------------------------------|
| handle                       | void*      | in     | ControllerPtr                                                                           |
| global_trans_idx       | int32_t    | in     | global transducer index                                                                 |
| return                       | int32_t    | -      | device index                                                                            |

##  AUTDTransPositionByGlobal (autd3capi)

Get the position of transducer specified by the global transducer index.
The `handle` is the controller created by `AUTDCreateController`.

| Argument name / return       | type             | in/out | description                                                                              |
|------------------------------|------------------|--------|-----------------------------------------------------------------------------------------|
| handle                       | void*      | in     | ControllerPtr                                                                           |
| global_trans_idx       | int32_t    | in     | global transducer index                                                                 |
| x                            | double*          | out    | x coordinate of transducer position                                                     |
| y                            | double*          | out    | y coordinate of transducer position                                                     |
| z                            | double*          | out    | z coordinate of transducer position                                                     |
| return                       | void       | -      | nothing                                                                                 |


##  AUTDTransPositionByLocal (autd3capi)

Get the position of transducer specified by the device index and local transducer index.
The `handle` is the controller created by `AUTDCreateController`.

| Argument name / return       | type             | in/out | description                                                                              |
|------------------------------|------------------|--------|-----------------------------------------------------------------------------------------|
| handle                       | void*      | in     | ControllerPtr                                                                           |
| device_idx             | int32_t    | in     | device index                                                                            |
| local_trans_idx        | int32_t    | in     | local transducer index                                                                  |
| x                            | double*          | out    | x coordinate of transducer position                                                     |
| y                            | double*          | out    | y coordinate of transducer position                                                     |
| z                            | double*          | out    | z coordinate of transducer position                                                     |
| return                      | void       | -      | nothing                                                                                 |

##  AUTDDeviceXDirection (autd3capi)

Get the x direction of device specified by the device index.
The `handle` is the controller created by `AUTDCreateController`.

| Argument name / return       | type             | in/out | description                                                                              |
|------------------------------|------------------|--------|-----------------------------------------------------------------------------------------|
| handle                       | void*      | in     | ControllerPtr                                                                           |
| device_idx             | int32_t    | in     | device index                                                                            |
| x                            | double*          | out    | x coordinate of device x-direction                                                      |
| y                            | double*          | out    | y coordinate of device x-direction                                                      |
| z                            | double*          | out    | z coordinate of device x-direction                                                      |
| return                      | void       | -      | nothing                                                                                 |

##  AUTDDeviceYDirection (autd3capi)

Get the y direction of device specified by the device index.
The `handle` is the controller created by `AUTDCreateController`.

| Argument name / return       | type             | in/out | description                                                                              |
|------------------------------|------------------|--------|-----------------------------------------------------------------------------------------|
| handle                       | void*      | in     | ControllerPtr                                                                           |
| device_idx             | int32_t    | in     | device index                                                                            |
| x                            | double*          | out    | x coordinate of device y-direction                                                      |
| y                            | double*          | out    | y coordinate of device y-direction                                                      |
| z                            | double*          | out    | z coordinate of device y-direction                                                      |
| return                      | void       | -      | nothing                                                                                 |

##  AUTDDeviceZDirection (autd3capi)

Get the z direction of device specified by the device index.
The `handle` is the controller created by `AUTDCreateController`.

| Argument name / return       | type             | in/out | description                                                                              |
|------------------------------|------------------|--------|-----------------------------------------------------------------------------------------|
| handle                       | void*      | in     | ControllerPtr                                                                           |
| device_idx             | int32_t    | in     | device index                                                                            |
| x                            | double*          | out    | x coordinate of device z-direction                                                      |
| y                            | double*          | out    | y coordinate of device z-direction                                                      |
| z                            | double*          | out    | z coordinate of device z-direction                                                      |
| return                      | void       | -      | nothing                                                                                 |

##  AUTDGetFirmwareInfoListPointer (autd3capi)

Get a pointer to the Firmware information list.
The `handle` is the controller created by `AUTDCreateController`.
The list created by this function should be released with `AUTDFreeFirmwareInfoListPointer` at the end.

The actual firmware information is retrieved with `AUTDGetFirmwareInfo`.

This function returns a value less than 0 if an error occurred.
You can get the error message with `AUTDGetLastError` if an error occurs.

| Argument name / return       | type             | in/out | description                                                                              |
|------------------------------|------------------|--------|-----------------------------------------------------------------------------------------|
| handle                       | void*      | in     | ControllerPtr                                                                           |
| out                          | void**     | out    | pointer to pointer to Firmware information list                                         |

##  AUTDGetFirmwareInfo (autd3capi)

Get firmware information.
Use `p_firm_info_list` created by `AUTDGetFirmwareInfoListPointer`.

`cpu_ver` and `fpga_ver` should be a pointer to a buffer, where the length of 128 is enough.

| Argument name / return       | type             | in/out | description                                                                              |
|------------------------------|------------------|--------|-----------------------------------------------------------------------------------------|
| p_firm_info_list       | void*      | in     | pointer to Firmware information list                                                    |
| index                        | int32_t    | in     | Firmware information index                                                              |
| cpu_ver                | char*            | out    | pointer to CPU version string                                                           |
| fpga_ver               | char*            | out    | pointer to FPGA version string                                                          |
| return                       | void       | -      | nothing                                                                                 |

##  AUTDFreeFirmwareInfoListPointer (autd3capi)

Release the firmware information list obtained with `AUTDGetFirmwareInfoListPointer`.

| Argument name / return       | type             | in/out | description                                                                              |
|------------------------------|------------------|--------|-----------------------------------------------------------------------------------------|
| p_firm_info_list       | void*      | in     | pointer to Firmware information list                                                    |
| return                       | void       | -      | nothing                                                                                 |

##  AUTDGainNull (autd3capi)

Create Gain::Null.

You must delete the created Gain with `AUTDDeleteGain` at the end.

| Argument name / return       | type             | in/out | description                                                                              |
|------------------------------|------------------|--------|-----------------------------------------------------------------------------------------|
| gain                   | void**     | out    | pointer to GainPtr                                                                      |
| return                       | void       | -      | nothing                                                                                 |

##  AUTDGainGrouped (autd3capi)

Create Gain::Grouped.

You must delete the created Gain with `AUTDDeleteGain` at the end.

| Argument name / return       | type             | in/out | description                                                                              |
|------------------------------|------------------|--------|-----------------------------------------------------------------------------------------|
| gain                   | void**     | out    | pointer to GainPtr                                                                      |
| return                       | void       | -      | nothing                                                                                 |

##  AUTDGainGroupedAdd (autd3capi)

Add a Gain to Gain::Grouped.
`grouped_gain` is the Gain created by `AUTDGainGrouped`. 
`id` is specified by `AUTDAddDevice` or `AUTDAddDeviceQuaternion`.

| Argument name / return       | type             | in/out | description                                                                              |
|------------------------------|------------------|--------|-----------------------------------------------------------------------------------------|
| grouped_gain           | void*      | in     | pointer to Grouped Gain                                                                 |
| id                           | int32_t    | in     | Groupe Id                                                                               |
| gain                   | void*      | in     | GainPtr                                                                                 |
| return                       | void       | -      | nothing                                                                                 |

##  AUTDGainFocalPoint (autd3capi)

Create Gain::FocalPoint.

You must delete the created Gain with `AUTDDeleteGain` at the end.

| Argument name / return       | type             | in/out | description                                                                              |
|------------------------------|------------------|--------|-----------------------------------------------------------------------------------------|
| gain                   | void**     | out    | pointer to GainPtr                                                                      |
| x                            | double     | in     | x coordinate of focal point                                                             |
| y                            | double     | in     | y coordinate of focal point                                                             |
| z                            | double     | in     | z coordinate of focal point                                                             |
| duty                         | uint8_t    | in     | duty ratio of Gain                                                                      |
| return                       | void       | -      | nothing                                                                                 |

##  AUTDGainBesselBeam (autd3capi)

Create Gain::Bessel.

You must delete the created Gain with `AUTDDeleteGain` at the end.

| Argument name / return       | type             | in/out | description                                                                              |
|------------------------------|------------------|--------|-----------------------------------------------------------------------------------------|
| gain                   | void**     | out    | pointer to GainPtr                                                                      |
| x                            | double     | in     | x coordinate of apex                                                                    |
| y                            | double     | in     | y coordinate of apex                                                                    |
| z                            | double     | in     | z coordinate of apex                                                                    |
| n_x                    | double     | in     | x coordinate of direction                                                               |
| n_y                    | double     | in     | y coordinate of direction                                                               |
| n_z                    | double     | in     | z coordinate of direction                                                               |
| theta_z                | double     | in     | angle between the side of the cone and the plane perpendicular to direction of the beam |
| duty                         | uint8_t    | in     | duty ratio of Gain                                                                      |
| return                       | void       | -      | nothing                                                                                 |

##  AUTDGainPlaneWave (autd3capi)

Create Gain::PlaneWave.

You must delete the created Gain with `AUTDDeleteGain` at the end.

| Argument name / return       | type             | in/out | description                                                                              |
|------------------------------|------------------|--------|-----------------------------------------------------------------------------------------|
| gain                   | void**     | out    | pointer to GainPtr                                                                      |
| n_x                    | double     | in     | x coordinate of direction                                                               |
| n_y                    | double     | in     | y coordinate of direction                                                               |
| n_z                    | double     | in     | z coordinate of direction                                                               |
| duty                         | uint8_t    | in     | duty ratio of Gain                                                                      |
| return                       | void       | -      | nothing                                                                                 |

##  AUTDGainCustom (autd3capi)

Create Gain::Custom.

You must delete the created Gain with `AUTDDeleteGain` at the end.

| Argument name / return       | type             | in/out | description                                                                              |
|------------------------------|------------------|--------|-----------------------------------------------------------------------------------------|
| gain                   | void**     | out    | pointer to GainPtr                                                                      |
| data                   | uint16_t*  | in     | pointer to data                                                                         |
| data_length            | int32_t    | in     | length of data                                                                          |
| return                       | void       | -      | nothing                                                                                 |

##  AUTDGainTransducerTest (autd3capi)

Create Gain::TransducerTest.

You must delete the created Gain with `AUTDDeleteGain` at the end.

| Argument name / return       | type             | in/out | description                                                                              |
|------------------------------|------------------|--------|-----------------------------------------------------------------------------------------|
| gain                   | void**     | out    | pointer to GainPtr                                                                      |
| idx                    | int32_t    | in     | global index of transducer                                                              |
| duty                   | uint8_t    | in     | duty ratio                                                                              |
| phase                  | uint8_t    | in     | phase                                                                                   |
| return                       | void       | -      | nothing                                                                                 |

##  AUTDDeleteGain (autd3capi)

Delete the Gain you created.

| Argument name / return       | type             | in/out | description                                                                              |
|------------------------------|------------------|--------|-----------------------------------------------------------------------------------------|
| gain                   | void*      | in     | GainPtr                                                                                 |
| return                       | void       | -      | nothing                                                                                 |

##  AUTDModulationStatic (autd3capi)

Create Modulation::Static.

You should delete the modulation you created with `AUTDDeleteModulation` at the end.

| Argument name / return       | type             | in/out | description                                                                              |
|------------------------------|------------------|--------|-----------------------------------------------------------------------------------------|
| mod                    | void**     | out    | ModulationPtr                                                                           |
| duty                   | uint8_t    | in     | duty ratio                                                                              |
| return                       | void       | -      | nothing                                                                                 |

##  AUTDModulationCustom (autd3capi)

Create Modulation::Custom.

You should delete the modulation you created with `AUTDDeleteModulation` at the end.

| Argument name / return       | type             | in/out | description                                                                              |
|------------------------------|------------------|--------|-----------------------------------------------------------------------------------------|
| mod                    | void**     | out    | ModulationPtr                                                                           |
| buf                    | uint8_t*   | in     | ModulationPtr buffer                                                                    |
| size                   | uint32_t   | in     | length to modulation buffer                                                             |
| return                       | void       | -      | nothing                                                                                 |

##  AUTDModulationSine (autd3capi)

Create Modulation::Sine.

You should delete the modulation you created with `AUTDDeleteModulation` at the end.

| Argument name / return       | type             | in/out | description                                                                              |
|------------------------------|------------------|--------|-----------------------------------------------------------------------------------------|
| mod                    | void**     | out    | ModulationPtr                                                                           |
| freq                   | int32_t    | in     | frequency                                                                               |
| amplitude              | double     | in     | amplitude of sin wave                                                                   |
| offset                 | double     | in     | offset of sin wave                                                                      |
| return                       | void       | -      | nothing                                                                                 |

##  AUTDModulationSinePressure (autd3capi)

Create Modulation::SinePressure.

You should delete the modulation you created with `AUTDDeleteModulation` at the end.

| Argument name / return       | type             | in/out | description                                                                              |
|------------------------------|------------------|--------|-----------------------------------------------------------------------------------------|
| mod                    | void**     | out    | ModulationPtr                                                                           |
| freq                   | int32_t    | in     | frequency                                                                               |
| amplitude              | double     | in     | amplitude of sin wave                                                                   |
| offset                 | double     | in     | offset of sin wave                                                                      |
| return                       | void       | -      | nothing                                                                                 |

##  AUTDModulationSineLegacy (autd3capi)

Create Modulation::SineLegacy.

You should delete the modulation you created with `AUTDDeleteModulation` at the end.

| Argument name / return       | type             | in/out | description                                                                              |
|------------------------------|------------------|--------|-----------------------------------------------------------------------------------------|
| mod                    | void**     | out    | ModulationPtr                                                                           |
| freq                   | double     | in     | frequency                                                                               |
| amplitude              | double     | in     | amplitude of sin wave                                                                   |
| offset                 | double     | in     | offset of sin wave                                                                      |
| return                       | void       | -      | nothing                                                                                 |

##  AUTDModulationSquare (autd3capi)

Create Modulation::Square.

You should delete the modulation you created with `AUTDDeleteModulation` at the end.

| Argument name / return       | type             | in/out | description                                                                              |
|------------------------------|------------------|--------|-----------------------------------------------------------------------------------------|
| mod                    | void**     | out    | ModulationPtr                                                                           |
| freq                   | int32_t    | in     | frequency                                                                               |
| low                    | uint8_t    | in     | duty ratio at low level                                                                 |
| high                   | uint8_t    | in     | duty ratio at high level                                                                |
| duty                   | double     | in     | duty ratio of the square wave, i.e., ratio of duration of high level to period          |
| return                       | void       | -      | nothing                                                                                 |


##  AUTDModulationSamplingFreqDiv (autd3capi)

Get sampling frequency division ratio of Modulation.

| Argument name / return       | type             | in/out | description                                                                              |
|------------------------------|------------------|--------|-----------------------------------------------------------------------------------------|
| mod                    | void*      | in     | ModulationPtr                                                                           |
| return                       | uint32_t   | -      | sampling frequency division ratio                                                       |

##  AUTDModulationSetSamplingFreqDiv (autd3capi)

Set sampling frequency division ratio of Modulation.

| Argument name / return       | type             | in/out | description                                                                              |
|------------------------------|------------------|--------|-----------------------------------------------------------------------------------------|
| mod                    | void*      | in     | ModulationPtr                                                                           |
| freq_div               | uint32_t   | in     | sampling frequency division ratio                                                       |
| return                       | void       | -      | nothing                                                                                 |

##  AUTDModulationSamplingFreq (autd3capi)

Get sampling frequency of Modulation.

| Argument name / return       | type             | in/out | description                                                                              |
|------------------------------|------------------|--------|-----------------------------------------------------------------------------------------|
| mod                    | void*      | in     | ModulationPtr                                                                           |
| return                       | double     | -      | sampling frequency                                                                      |

##  AUTDDeleteModulation (autd3capi)

Delete the Modulation you created.

| Argument name / return       | type             | in/out | description                                                                              |
|------------------------------|------------------|--------|-----------------------------------------------------------------------------------------|
| mod                    | void*      | in     | ModulationPtr                                                                           |
| return                       | void       | -      | nothing                                                                                 |

##  AUTDSequence (autd3capi)

Create PointSequence.
You have to delete the created PointSequence with `AUTDDeleteSequence` at the end.

| Argument name / return       | type             | in/out | description                                                                              |
|------------------------------|------------------|--------|-----------------------------------------------------------------------------------------|
| out                    | void**     | out    | pointer to PointSequencePtr                                                             |
| return                       | void       | -      | nothing                                                                                 |

##  AUTDGainSequence (autd3capi)

Create GainSequence.

You have to delete the created GainSequence with `AUTDDeleteSequence` at the end.

| Argument name / return       | type             | in/out | description                                                                              |
|------------------------------|------------------|--------|-----------------------------------------------------------------------------------------|
| out                    | void**     | out    | pointer to GainSequencePtr                                                              |
| gain_mode              | uint16_t   | in     | gain mode of GainSequence                                                               |
| return                       | void       | -      | nothing                                                                                 |

##  AUTDSequenceAddPoint (autd3capi)

Add control point to PointSequence.

`seq` is a PointSequence created with `AUTDSequence`.

This function returns false if it fails.
If it returns false, you can get the error message with `AUTDGetLastError`.

| Argument name / return       | type             | in/out | description                                                                              |
|------------------------------|------------------|--------|-----------------------------------------------------------------------------------------|
| seq                    | void*      | in     | GainSequencePtr                                                                         |
| x                      | double     | in     | x coordinate of point                                                                   |
| y                      | double     | in     | y coordinate of point                                                                   |
| z                      | double     | in     | z coordinate of point                                                                   |
| duty                   | uint8_t    | in     | duty ratio of point                                                                     |
| return                       | bool       | -      | true if success                                                                         |

##  AUTDSequenceAddPoints (autd3capi)

Add control points to PointSequence.

`seq` is a PointSequence created with `AUTDSequence`.

`points` is a pointer to an array of length `points_size` $\times 3$,
where the data is stored in x\[0\], y\[0\], z\[0\], x\[1\], y\[1\],
z\[1\], ... order.
If `duties_size` is less than `points_size`, the missing data will be filled with duty=0xFF.
If `duties_size` is greater than `points_size`, the excess is ignored.

This function returns false if it fails.
If it returns false, you can get the error message with `AUTDGetLastError`.

| Argument name / return       | type             | in/out | description                                                                              |
|------------------------------|------------------|--------|-----------------------------------------------------------------------------------------|
| seq                    | void*      | in     | GainSequencePtr                                                                         |
| points                 | double*    | in     | pointer to points array                                                                 |
| points_size            | uint64_t   | in     | length of points array                                                                  |
| duties                 | double*    | in     | pointer to duties array                                                                 |
| duties_size            | uint64_t   | in     | length of duties array                                                                  |
| return                       | bool       | -      | true if success                                                                         |

##  AUTDSequenceAddGain (autd3capi)

Add Gain to GainSequence.

`seq` is a GainSequence created with `AUTDGainSequence`.

This function returns false if it fails.
If it returns false, you can get the error message with `AUTDGetLastError`.

| Argument name / return       | type             | in/out | description                                                                              |
|------------------------------|------------------|--------|-----------------------------------------------------------------------------------------|
| seq                    | void*      | in     | GainSequencePtr                                                                         |
| gain                   | void*      | in     | GainPtr                                                                                 |
| return                       | bool       | -      | true if success                                                                         |

##  AUTDSequenceSetFreq (autd3capi)

Set frequency of Sequence.

`seq` is a PointSequence created with `AUTDSequence` or a GainSequence created with `AUTDGainSequence`.

This function returns actual frequency.

| Argument name / return       | type             | in/out | description                                                                              |
|------------------------------|------------------|--------|-----------------------------------------------------------------------------------------|
| seq                    | void*      | in     | SequencePtr                                                                             |
| freq                   | double     | in     | frequency                                                                               |
| return                       | double     | -      | actual frequency                                                                        |

##  AUTDSequenceFreq (autd3capi)

Get frequency of Sequence.

`seq` is a PointSequence created with `AUTDSequence` or a GainSequence created with `AUTDGainSequence`.

| Argument name / return       | type             | in/out | description                                                                              |
|------------------------------|------------------|--------|-----------------------------------------------------------------------------------------|
| seq                    | void*      | in     | SequencePtr                                                                             |
| return                       | double     | -      | actual frequency                                                                        |

##  AUTDSequencePeriod (autd3capi)

Get cycle of Sequence in unit of μs.

`seq` is a PointSequence created with `AUTDSequence` or a GainSequence created with `AUTDGainSequence`.

| Argument name / return       | type             | in/out | description                                                                              |
|------------------------------|------------------|--------|-----------------------------------------------------------------------------------------|
| seq                    | void*      | in     | SequencePtr                                                                             |
| return                       | uint32_t   | -      | period in μs                                                   |

##  AUTDSequenceSamplingPeriod (autd3capi)

Get sampling cycle of Sequence in unit of μs.

`seq` is a PointSequence created with `AUTDSequence` or a GainSequence created with `AUTDGainSequence`.

| Argument name / return       | type             | in/out | description                                                                              |
|------------------------------|------------------|--------|-----------------------------------------------------------------------------------------|
| seq                    | void*      | in     | SequencePtr                                                                             |
| return                       | uint32_t   | -      | sampling period in μs                                          |

##  AUTDSequenceSamplingFreq (autd3capi)

Get sampling frequency of Sequence.

`seq` is a PointSequence created with `AUTDSequence` or a GainSequence created with `AUTDGainSequence`.

| Argument name / return       | type             | in/out | description                                                                              |
|------------------------------|------------------|--------|-----------------------------------------------------------------------------------------|
| seq                    | void*      | in     | SequencePtr                                                                             |
| return                       | double     | -      | sampling freqyency                                                                      |

##  AUTDSequenceSamplingFreqDiv (autd3capi)

Get sampling frequency division ratio of Sequence.

`seq` is a PointSequence created with `AUTDSequence` or a GainSequence created with `AUTDGainSequence`.

| Argument name / return       | type             | in/out | description                                                                              |
|------------------------------|------------------|--------|-----------------------------------------------------------------------------------------|
| seq                    | void*      | in     | SequencePtr                                                                             |
| return                       | uint32_t   | -      | sampling freqyency division ratio                                                       |

##  AUTDSequenceSetSamplingFreqDiv (autd3capi)

Set sampling frequency division ratio of Sequence.

`seq` is a PointSequence created with `AUTDSequence` or a GainSequence created with `AUTDGainSequence`.

| Argument name / return       | type             | in/out | description                                                                              |
|------------------------------|------------------|--------|-----------------------------------------------------------------------------------------|
| seq                    | void*      | in     | SequencePtr                                                                             |
| freq_div               | uint32_t   | in     | sampling freqyency division ratio                                                       |
| return                       | void       | -      | nothing                                                                                 |

##  AUTDCircumSequence (autd3capi)

Create PointSequence on a circumference.

You have to delete the created PointSequence with `AUTDDeleteSequence` at the end.

| Argument name / return       | type             | in/out | description                                                                              |
|------------------------------|------------------|--------|-----------------------------------------------------------------------------------------|
| out                    | void**     | out    | pointer to PointSequencePtr                                                             |
| x                      | double     | in     | x coordinate of circumference                                                           |
| y                      | double     | in     | y coordinate of circumference                                                           |
| z                      | double     | in     | z coordinate of circumference                                                           |
| nx                     | double     | in     | x coordinate of normal of circumference                                                 |
| ny                     | double     | in     | y coordinate of normal of circumference                                                 |
| nz                     | double     | in     | z coordinate of normal of circumference                                                 |
| radius                 | double     | in     | radius of circumference                                                                 |
| n                      | uint64_t   | in     | number of sampling points                                                               |
| return                       | void       | -      | nothing                                                                                 |

##  AUTDDeleteSequence (autd3capi)

Delete Sequence you created.

| Argument name / return       | type             | in/out | description                                                                              |
|------------------------------|------------------|--------|-----------------------------------------------------------------------------------------|
| seq                    | void*      | in     | SequencePtr                                                                             |
| return                       | void       | -      | nothing                                                                                 |

##  AUTDStop (autd3capi)

Stop outputting.
One of the send functions.

The `handle` is the controller created by `AUTDCreateController`.

This function returns a value less than 0 if an error occurred.
If an error occurs, you can get the error message with `AUTDGetLastError`.
Also, if the check ack flag is on and the return value is greater than 0, it guarantees that the data has been processed by the device.

| Argument name / return       | type             | in/out | description                                                                              |
|------------------------------|------------------|--------|-----------------------------------------------------------------------------------------|
| handle                 | void*      | in     | ControllerPtr                                                                           |
| return                       | int32_t    | -      | if $>0$, it guarantees devices have processed data. if $<0$, error ocurred.             |


##  AUTDPause (autd3capi)

Pause outputting.
One of the send functions. 
The output can be resumed with `AUTDResume`.

The `handle` is the controller created by `AUTDCreateController`.

This function returns a value less than 0 if an error occurred.
If an error occurs, you can get the error message with `AUTDGetLastError`.
Also, if the check ack flag is on and the return value is greater than 0, it guarantees that the data has been processed by the device.

| Argument name / return       | type             | in/out | description                                                                              |
|------------------------------|------------------|--------|-----------------------------------------------------------------------------------------|
| handle                 | void*      | in     | ControllerPtr                                                                           |
| return                       | int32_t    | -      | if $>0$, it guarantees devices have processed data. if $<0$, error ocurred.             |


##  AUTDResume (autd3capi)

Resume outputting.
One of the send functions.

The `handle` is the controller created by `AUTDCreateController`.

This function returns a value less than 0 if an error occurred.
If an error occurs, you can get the error message with `AUTDGetLastError`.
Also, if the check ack flag is on and the return value is greater than 0, it guarantees that the data has been processed by the device.

| Argument name / return       | type             | in/out | description                                                                              |
|------------------------------|------------------|--------|-----------------------------------------------------------------------------------------|
| handle                 | void*      | in     | ControllerPtr                                                                           |
| return                       | int32_t    | -      | if $>0$, it guarantees devices have processed data. if $<0$, error ocurred.             |

##  AUTDSendGain (autd3capi)

Send Gain.
One of the send functions.

The `handle` is the controller created by `AUTDCreateController`.

If `check_ack` is true, the function waits until the data is processed by the device.

This function returns a value less than 0 if an error occurred.
If an error occurs, you can get the error message with `AUTDGetLastError`.
Also, if the check ack flag is on and the return value is greater than 0, it guarantees that the data has been processed by the device.

| Argument name / return       | type             | in/out | description                                                                              |
|------------------------------|------------------|--------|-----------------------------------------------------------------------------------------|
| handle                 | void*      | in     | ControllerPtr                                                                           |
| gain                   | void*      | in     | GainPtr                                                                                 |
| return                       | int32_t    | -      | if $>0$, it guarantees devices have processed data. if $<0$, error ocurred.             |

##  AUTDSendModulation (autd3capi)

Send Modulation. 
One of the send functions.

The `handle` is the controller created by `AUTDCreateController`.

If `check_ack` is true, the function waits until the data is processed by the device.

This function returns a value less than 0 if an error occurred.
If an error occurs, you can get the error message with `AUTDGetLastError`.
Also, if the check ack flag is on and the return value is greater than 0, it guarantees that the data has been processed by the device.

| Argument name / return       | type             | in/out | description                                                                              |
|------------------------------|------------------|--------|-----------------------------------------------------------------------------------------|
| handle                 | void*      | in     | ControllerPtr                                                                           |
| mod                    | void*      | in     | ModulationPtr                                                                           |
| return                       | int32_t    | -      | if $>0$, it guarantees devices have processed data. if $<0$, error ocurred.             |

##  AUTDSendGainModulation (autd3capi)

Send Gain and Modulation. 
One of the send functions.

The `handle` is the controller created by `AUTDCreateController`.

If `check_ack` is true, the function waits until the data is processed by the device.

This function returns a value less than 0 if an error occurred.
If an error occurs, you can get the error message with `AUTDGetLastError`.
Also, if the check ack flag is on and the return value is greater than 0, it guarantees that the data has been processed by the device.

| Argument name / return       | type             | in/out | description                                                                              |
|------------------------------|------------------|--------|-----------------------------------------------------------------------------------------|
| handle                 | void*      | in     | ControllerPtr                                                                           |
| gain                   | void*      | in     | GainPtr                                                                                 |
| mod                    | void*      | in     | ModulationPtr                                                                           |
| return                       | int32_t    | -      | if $>0$, it guarantees devices have processed data. if $<0$, error ocurred.             |

##  AUTDSendSequenceModulation (autd3capi)

Send PointSequence and Modulation. 
One of the send functions.

The `handle` is the controller created by `AUTDCreateController`.

If `check_ack` is true, the function waits until the data is processed by the device.

This function returns a value less than 0 if an error occurred.
If an error occurs, you can get the error message with `AUTDGetLastError`.
Also, if the check ack flag is on and the return value is greater than 0, it guarantees that the data has been processed by the device.

| Argument name / return       | type             | in/out | description                                                                              |
|------------------------------|------------------|--------|-----------------------------------------------------------------------------------------|
| handle                 | void*      | in     | ControllerPtr                                                                           |
| seq                    | void*      | in     | PointSequencePtr                                                                        |
| mod                    | void*      | in     | ModulationPtr                                                                           |
| return                       | int32_t    | -      | if $>0$, it guarantees devices have processed data. if $<0$, error ocurred.             |

##  AUTDSendGainSequenceModulation (autd3capi)

Send GainSequence and Modulation. 
One of the send functions.

The `handle` is the controller created by `AUTDCreateController`.

If `check_ack` is true, the function waits until the data is processed by the device.

This function returns a value less than 0 if an error occurred.
If an error occurs, you can get the error message with `AUTDGetLastError`.
Also, if the check ack flag is on and the return value is greater than 0, it guarantees that the data has been processed by the device.

| Argument name / return       | type             | in/out | description                                                                              |
|------------------------------|------------------|--------|-----------------------------------------------------------------------------------------|
| handle                 | void*      | in     | ControllerPtr                                                                           |
| seq                    | void*      | in     | GainSequencePtr                                                                         |
| mod                    | void*      | in     | ModulationPtr                                                                           |
| return                       | int32_t    | -      | if $>0$, it guarantees devices have processed data. if $<0$, error ocurred.             |

##  AUTDSTMController (autd3capi)

Get STMController from Controller.

The `handle` is the controller created by `AUTDCreateController`.

The use of handle is prohibited between the call to this function and the call to `AUTDFinishSTM`.

| Argument name / return       | type             | in/out | description                                                                              |
|------------------------------|------------------|--------|-----------------------------------------------------------------------------------------|
| out                    | void**     | out    | pointer to STMControllerPtr                                                             |
| handle                 | void*      | out    | ControllerPtr                                                                        |
| return                       | void       | -      | nothing                                                                                 |

##  AUTDAddSTMGain (autd3capi)

Add Gain to STMController.

The `handle` is the stm controller created by `AUTDSTMController`.

This function returns false if it fails.
If it returns false, you can get the error message with `AUTDGetLastError`.

| Argument name / return       | type             | in/out | description                                                                              |
|------------------------------|------------------|--------|-----------------------------------------------------------------------------------------|
| handle                 | void*      | in     | STMControllerPtr                                                                        |
| gain                   | void*      | in     | GainPtr                                                                                 |
| return                       | bool       | -      | true if success                                                                         |

##  AUTDStartSTM (autd3capi)

Start STM.

The `handle` is the stm controller created by `AUTDSTMController`.

This function returns false if it fails.
If it returns false, you can get the error message with `AUTDGetLastError`.

| Argument name / return       | type             | in/out | description                                                                              |
|------------------------------|------------------|--------|-----------------------------------------------------------------------------------------|
| handle                 | void*      | in     | STMControllerPtr                                                                        |
| freq                   | double     | in     | frequency                                                                               |
| return                       | bool       | -      | true if success                                                                         |

##  AUTDStopSTM (autd3capi)

Stop STM.

The `handle` is the stm controller created by `AUTDSTMController`.

This function returns false if it fails.
If it returns false, you can get the error message with `AUTDGetLastError`.

| Argument name / return       | type             | in/out | description                                                                              |
|------------------------------|------------------|--------|-----------------------------------------------------------------------------------------|
| handle                 | void*      | in     | STMControllerPtr                                                                        |
| return                       | bool       | -      | true if success                                                                         |

##  AUTDFinishSTM (autd3capi)

Finish STM.

The `handle` is the stm controller created by `AUTDSTMController`.

This function returns false if it fails.
If it returns false, you can get the error message with `AUTDGetLastError`.

| Argument name / return       | type             | in/out | description                                                                              |
|------------------------------|------------------|--------|-----------------------------------------------------------------------------------------|
| handle                 | void*      | in     | STMControllerPtr                                                                        |
| return                       | bool       | -      | true if success                                                                         |

##  AUTDEigen3Backend (autd3capi-holo-gain)

Create Eigen Backend.

You should delete the created Backend with `AUTDDeleteBackend` in the end.

| Argument name / return       | type             | in/out | description                                                                              |
|------------------------------|------------------|--------|-----------------------------------------------------------------------------------------|
| out                    | void**     | out    | pointer to BackendPtr                                                                   |
| return                       | void       | -      | nothing                                                                                 |

##  AUTDDeleteBackend (autd3capi-holo-gain)

Delete Backend.

| Argument name / return       | type             | in/out | description                                                                              |
|------------------------------|------------------|--------|-----------------------------------------------------------------------------------------|
| backend                | void*      | in     | BackendPtr                                                                              |
| return                       | void       | -      | nothing                                                                                 |

##  AUTDGainHoloSDP (autd3capi-holo-gain)

Create holo::SDP.

`points` is a pointer to an array of length `size` $\times 3$, where the focal points are stored in the following order: x\[0\], y\[0\], z\[0\], x\[1\], ...
`amps` is a pointer to an array of length `size` and specifies the intensity of foci.

You must delete the created Gain with `AUTDDeleteGain` at the end.

| Argument name / return       | type             | in/out | description                                                                              |
|------------------------------|------------------|--------|-----------------------------------------------------------------------------------------|
| gain                   | void**     | out    | pointer to GainPtr                                                                      |
| backend                | void*      | in     | BackendPtr                                                                              |
| points                 | double*          | in     | pointer to foci array                                                                   |
| amps                   | double*          | in     | pointer to amplitudes array                                                             |
| size                   | int32_t    | in     | number of foci                                                                          |
| alpha                  | double     | in     | parameter                                                                               |
| repeat                 | uint64_t   | in     | parameter                                                                               |
| normalize              | bool       | in     | parameter                                                                               |
| return                       | void       | -      | nothing                                                                                 |

##  AUTDGainHoloEVD (autd3capi-holo-gain)

Create holo::EVD.

`points` is a pointer to an array of length `size` $\times 3$, where the focal points are stored in the following order: x\[0\], y\[0\], z\[0\], x\[1\], ...
`amps` is a pointer to an array of length `size` and specifies the intensity of foci.

You must delete the created Gain with `AUTDDeleteGain` at the end.

| Argument name / return       | type             | in/out | description                                                                              |
|------------------------------|------------------|--------|-----------------------------------------------------------------------------------------|
| gain                   | void**     | out    | pointer to GainPtr                                                                      |
| backend                | void*      | in     | BackendPtr                                                                              |
| points                 | double*          | in     | pointer to foci array                                                                   |
| amps                   | double*          | in     | pointer to amplitudes array                                                             |
| size                   | int32_t    | in     | number of foci                                                                          |
| gamma                  | double     | in     | parameter                                                                               |
| normalize              | bool       | in     | parameter                                                                               |
| return                       | void       | -      | nothing                                                                                 |

##  AUTDGainHoloNaive (autd3capi-holo-gain)

Create holo::Naive.

`points` is a pointer to an array of length `size` $\times 3$, where the focal points are stored in the following order: x\[0\], y\[0\], z\[0\], x\[1\], ...
`amps` is a pointer to an array of length `size` and specifies the intensity of foci.

You must delete the created Gain with `AUTDDeleteGain` at the end.

| Argument name / return       | type             | in/out | description                                                                              |
|------------------------------|------------------|--------|-----------------------------------------------------------------------------------------|
| gain                   | void**     | out    | pointer to GainPtr                                                                      |
| backend                | void*      | in     | BackendPtr                                                                              |
| points                 | double*          | in     | pointer to foci array                                                                   |
| amps                   | double*          | in     | pointer to amplitudes array                                                             |
| size                   | int32_t    | in     | number of foci                                                                          |
| return                       | void       | -      | nothing                                                                                 |

##  AUTDGainHoloGS (autd3capi-holo-gain)

Create holo::GS.

`points` is a pointer to an array of length `size` $\times 3$, where the focal points are stored in the following order: x\[0\], y\[0\], z\[0\], x\[1\], ...
`amps` is a pointer to an array of length `size` and specifies the intensity of foci.

You must delete the created Gain with `AUTDDeleteGain` at the end.

| Argument name / return       | type             | in/out | description                                                                              |
|------------------------------|------------------|--------|-----------------------------------------------------------------------------------------|
| gain                   | void**     | out    | pointer to GainPtr                                                                      |
| backend                | void*      | in     | BackendPtr                                                                              |
| points                 | double*          | in     | pointer to foci array                                                                   |
| amps                   | double*          | in     | pointer to amplitudes array                                                             |
| size                   | int32_t    | in     | number of foci                                                                          |
| repeat                 | uint64_t   | in     | parameter                                                                               |
| return                       | void       | -      | nothing                                                                                 |

##  AUTDGainHoloGSPAT (autd3capi-holo-gain)

Create holo::GSPAT.

`points` is a pointer to an array of length `size` $\times 3$, where the focal points are stored in the following order: x\[0\], y\[0\], z\[0\], x\[1\], ...
`amps` is a pointer to an array of length `size` and specifies the intensity of foci.

You must delete the created Gain with `AUTDDeleteGain` at the end.

| Argument name / return       | type             | in/out | description                                                                              |
|------------------------------|------------------|--------|-----------------------------------------------------------------------------------------|
| gain                   | void**     | out    | pointer to GainPtr                                                                      |
| backend                | void*      | in     | BackendPtr                                                                              |
| points                 | double*          | in     | pointer to foci array                                                                   |
| amps                   | double*          | in     | pointer to amplitudes array                                                             |
| size                   | int32_t    | in     | number of foci                                                                          |
| repeat                 | uint64_t   | in     | parameter                                                                               |
| return                       | void       | -      | nothing                                                                                 |

##  AUTDGainHoloLM (autd3capi-holo-gain)

Create holo::LM.

`points` is a pointer to an array of length `size` $\times 3$, where the focal points are stored in the following order: x\[0\], y\[0\], z\[0\], x\[1\], ...
`amps` is a pointer to an array of length `size` and specifies the intensity of foci.

You must delete the created Gain with `AUTDDeleteGain` at the end.

| Argument name / return       | type             | in/out | description                                                                              |
|------------------------------|------------------|--------|-----------------------------------------------------------------------------------------|
| gain                   | void**     | out    | pointer to GainPtr                                                                      |
| backend                | void*      | in     | BackendPtr                                                                              |
| points                 | double*          | in     | pointer to foci array                                                                   |
| amps                   | double*          | in     | pointer to amplitudes array                                                             |
| size                   | int32_t    | in     | number of foci                                                                          |
| eps_1                  | double     | in     | parameter                                                                               |
| eps_2                  | double     | in     | parameter                                                                               |
| tau                    | double     | in     | parameter                                                                               |
| k_max                  | uint64_t   | in     | parameter                                                                               |
| initial                | double*    | in     | parameter                                                                               |
| initial_size           | int32_t    | in     | length of initial                                                                       |
| return                       | void       | -      | nothing                                                                                 |

##  AUTDGainHoloGaussNewton (autd3capi-holo-gain)

Create holo::GaussNewton.

`points` is a pointer to an array of length `size` $\times 3$, where the focal points are stored in the following order: x\[0\], y\[0\], z\[0\], x\[1\], ...
`amps` is a pointer to an array of length `size` and specifies the intensity of foci.

You must delete the created Gain with `AUTDDeleteGain` at the end.

| Argument name / return       | type             | in/out | description                                                                              |
|------------------------------|------------------|--------|-----------------------------------------------------------------------------------------|
| gain                   | void**     | out    | pointer to GainPtr                                                                      |
| backend                | void*      | in     | BackendPtr                                                                              |
| points                 | double*          | in     | pointer to foci array                                                                   |
| amps                   | double*          | in     | pointer to amplitudes array                                                             |
| size                   | int32_t    | in     | number of foci                                                                          |
| eps_1                  | double     | in     | parameter                                                                               |
| eps_2                  | double     | in     | parameter                                                                               |
| k_max                  | uint64_t   | in     | parameter                                                                               |
| initial                | double*    | in     | parameter                                                                               |
| initial_size           | int32_t    | in     | length of initial                                                                       |
| return                       | void       | -      | nothing                                                                                 |

##  AUTDGainHoloGradientDescent (autd3capi-holo-gain)

Create holo::GradientDescent.

`points` is a pointer to an array of length `size` $\times 3$, where the focal points are stored in the following order: x\[0\], y\[0\], z\[0\], x\[1\], ...
`amps` is a pointer to an array of length `size` and specifies the intensity of foci.

You must delete the created Gain with `AUTDDeleteGain` at the end.

| Argument name / return       | type             | in/out | description                                                                              |
|------------------------------|------------------|--------|-----------------------------------------------------------------------------------------|
| gain                   | void**     | out    | pointer to GainPtr                                                                      |
| backend                | void*      | in     | BackendPtr                                                                              |
| points                 | double*          | in     | pointer to foci array                                                                   |
| amps                   | double*          | in     | pointer to amplitudes array                                                             |
| size                   | int32_t    | in     | number of foci                                                                          |
| eps                    | double     | in     | parameter                                                                               |
| step                   | double     | in     | parameter                                                                               |
| k_max                  | uint64_t   | in     | parameter                                                                               |
| initial                | double*    | in     | parameter                                                                               |
| initial_size           | int32_t    | in     | length of initial                                                                       |
| return                       | void       | -      | nothing                                                                                 |

##  AUTDGainHoloAPO (autd3capi-holo-gain)

Create holo::APO.

`points` is a pointer to an array of length `size` $\times 3$, where the focal points are stored in the following order: x\[0\], y\[0\], z\[0\], x\[1\], ...
`amps` is a pointer to an array of length `size` and specifies the intensity of foci.

You must delete the created Gain with `AUTDDeleteGain` at the end.

| Argument name / return       | type             | in/out | description                                                                              |
|------------------------------|------------------|--------|-----------------------------------------------------------------------------------------|
| gain                   | void**     | out    | pointer to GainPtr                                                                      |
| backend                | void*      | in     | BackendPtr                                                                              |
| points                 | double*          | in     | pointer to foci array                                                                   |
| amps                   | double*          | in     | pointer to amplitudes array                                                             |
| size                   | int32_t    | in     | number of foci                                                                          |
| eps                    | double     | in     | parameter                                                                               |
| lambda                 | double     | in     | parameter                                                                               |
| k_max                  | uint64_t   | in     | parameter                                                                               |
| return                       | void       | -      | nothing                                                                                 |

##  AUTDGainHoloGreedy (autd3capi-holo-gain)

Create holo::Greedy.

`points` is a pointer to an array of length `size` $\times 3$, where the focal points are stored in the following order: x\[0\], y\[0\], z\[0\], x\[1\], ...
`amps` is a pointer to an array of length `size` and specifies the intensity of foci.

You must delete the created Gain with `AUTDDeleteGain` at the end.

| Argument name / return       | type             | in/out | description                                                                              |
|------------------------------|------------------|--------|-----------------------------------------------------------------------------------------|
| gain                   | void**     | out    | pointer to GainPtr                                                                      |
| backend                | void*      | in     | BackendPtr                                                                              |
| points                 | double*          | in     | pointer to foci array                                                                   |
| amps                   | double*          | in     | pointer to amplitudes array                                                             |
| size                   | int32_t    | in     | number of foci                                                                          |
| phase_div              | int32_t    | in     | parameter                                                                               |
| return                       | void       | -      | nothing                                                                                 |

##  AUTDModulationRawPCM (autd3capi-from-file-modulation)

Create modulation::RawPCM.

You should delete the modulation you created with `AUTDDeleteModulation` at the end.

| Argument name / return       | type             | in/out | description                                                                              |
|------------------------------|------------------|--------|-----------------------------------------------------------------------------------------|
| mod                    | void**     | out    | pointer to ModulationPtr                                                                |
| filename               | char*            | in     | pointer to filename string                                                              |
| sampling_freq          | double     | in     | sampling frequency of PCM                                                               |
| mod_sampling_freq_div  | uint16_t   | in     | Sampling frequency division                                                             |
| return                       | void       | -      | nothing                                                                                 |

##  AUTDModulationWav (autd3capi-from-file-modulation)

Create modulation::Wav.

You should delete the modulation you created with `AUTDDeleteModulation` at the end.

| Argument name / return       | type             | in/out | description                                                                              |
|------------------------------|------------------|--------|-----------------------------------------------------------------------------------------|
| mod                    | void**     | out    | pointer to ModulationPtr                                                                |
| filename               | char*            | in     | pointer to filename string                                                              |
| mod_sampling_freq_div  | uint16_t   | in     | Sampling frequency division                                                             |
| return                       | void       | -      | nothing                                                                                 |

##  AUTDLinkTwinCAT (autd3capi-twincat-link)

Create link::TwinCAT.

| Argument name / return       | type             | in/out | description                                                                              |
|------------------------------|------------------|--------|-----------------------------------------------------------------------------------------|
| out                    | void**     | out    | pointer to LinkPtr                                                                      |
| return                       | void       | -      | nothing                                                                                 |

##  AUTDLinkRemoteTwinCAT (autd3capi-remote-twincat-link)

Create link::RemoteTwinCAT.

| Argument name / return       | type             | in/out | description                                                                              |
|------------------------------|------------------|--------|-----------------------------------------------------------------------------------------|
| out                    | void**     | out    | pointer to LinkPtr                                                                      |
| remote_ip_addr         | char*            | in     | pointer to remote ip address                                                            |
| remote_ams_net_id      | char*            | in     | pointer to remote ams net id                                                            |
| local_ams_net_id       | char*            | in     | pointer to local ams net id                                                             |
| return                       | void       | -      | nothing                                                                                 |

##  AUTDGetAdapterPointer (autd3capi-soem-link)

Get a pointer to the EtherCAT adapter information list.

Get the EtherCAT adapter information with `AUTDGetAdapter`.

The retrieved pointer has to be finally released with `AUTDFreeAdapterPointer`.

This function returns the size of the retrieved list.

| Argument name / return       | type             | in/out | description                                                                              |
|------------------------------|------------------|--------|-----------------------------------------------------------------------------------------|
| out                    | void**     | out    | pointer to pointer to adapter information list                                          |
| return                       | int32_t    | -      | length of EtherCAT adapter information list                                             |

##  AUTDGetAdapter (autd3capi-soem-link)

Get EtherCAT adapter information.

`p_adapter` is a pointer created by `AUTDGetAdapterPointer`.

`desc` and `name` are pointers to a buffer, where the length of 128 is sufficient.

| Argument name / return       | type             | in/out | description                                                                              |
|------------------------------|------------------|--------|-----------------------------------------------------------------------------------------|
| p_adapter              | void*      | in     | pointer to EtherCAT adapter information list                                            |
| index                  | int32_t    | in     | adapter index                                                                           |
| desc                   | char*      | in     | adapter description                                                                     |
| name                   | char*      | in     | adapter name                                                                            |
| return                       | void       | -      | nothing                                                                                 |

##  AUTDFreeAdapterPointer (autd3capi-soem-link)

Release pointer created by `AUTDGetAdapterPointer`.

| Argument name / return       | type             | in/out | description                                                                              |
|------------------------------|------------------|--------|-----------------------------------------------------------------------------------------|
| p_adapter              | void*      | in     | pointer to EtherCAT adapter information list                                            |
| return                       | void       | -      | nothing                                                                                 |

##  AUTDLinkSOEM (autd3capi-soem-link)

Create link::SOEM.

`ifname` is a interface name, which can be obtained by `AUTDGetAdapter`.

| Argument name / return       | type             | in/out | description                                                                              |
|------------------------------|------------------|--------|-----------------------------------------------------------------------------------------|
| out                    | void**     | out    | pointer to LinkPtr                                                                      |
| ifname                 | int32_t    | in     | number of devices                                                                       |
| cycle_ticks            | uint32_t   | in     | cycle ticks                                                                             |
| return                       | void       | -      | nothing                                                                                 |

##  AUTDSetSOEMOnLost (autd3capi-soem-link)

Set OnLostCallback to SOEM link.

`link` is a link created by `AUTDLinkSOEM`.

`ErrorHandler` is defined as `void (*)(const char*)`, and handler will be called with error message when error occurred.

| Argument name / return       | type             | in/out | description                                                                              |
|------------------------------|------------------|--------|-----------------------------------------------------------------------------------------|
| link                   | void*      | in     | SOEM LinkPtr                                                                            |
| handler                | ErrorHandler     | in     | error handler                                                                           |
| return                       | void       | -      | nothing                                                                                 |

##  AUTDLinkEmulator (autd3capi-emulator-link)

Create link::Emulator.

`cnt` is a controller created by `AUTDCreateController`.

| Argument name / return       | type             | in/out | description                                                                              |
|------------------------------|------------------|--------|-----------------------------------------------------------------------------------------|
| out                    | void**     | out    | pointer to LinkPtr                                                                      |
| port                   | uint16_t   | in     | number of devices                                                                       |
| cnt                    | void*      | in     | ControllerPtr                                                                           |
| return                       | void       | -      | nothing                                                                                 |
