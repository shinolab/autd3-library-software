# Controller

In this section, we introduce some other functions of the Controller class.

## Output enable

Configure the output enable settings.
```cpp
  autd->output_enable() = false;
```
The output of the FPGA is the logical product of this flag.

The flag will be actually updated after calling one of the [Send functions](#send-functions).

## Silent mode

In AM and Spatio-Temporal Modulation, noise is generated while the phase/amplitude changes abruptly.
SDK provides a flag to suppress this noise.
```cpp
  autd->silent_mode() = true;
```
When this flag is set to on, a low-pass filter is applied to the phase/amplitude data inside the device, smoothing the phase/amplitude changes and suppressing noise [suzuki2020].

The flag will be actually updated after calling one of the [Send functions](#send-functions).

## Check Ack

If the `check_ack` flag is set to on, when sending data to the device, SDK will check whether the sent data has been processed by the device or not.
```cpp
  autd->check_ack() = true;
```
If `check_ack` is `true`, functions that send data to the device ([Send functions](#send-functions)) will return whether the sent data has been properly processed by the device or not.

Basically, you have no problem with this flag off, but if you want to send data certainly, turn this flag on.
Note that the execution time of [Send functions](#send-functions) will increase if you turn this flag on.

## Force fan

AUTD3 devices are equipped with a thermal sensor, which automatically starts a fan when the temperature becomes too high.
The `force_fan` flag is a flag to force the fan to start.

The flag will be actually updated after calling one of the [Send functions](#send-functions).

```cpp
  autd->force_fan() = true;
```

Note that the fan can be forced on, but it cannot be forced off.

## Read FPGA info

If you turn on the `reads_fpga_info` flag, the device will return the FPGA status.

The flag will be actually updated after calling one of the [Send functions](#send-functions).

The status of the FPGA can be obtained with the `fpga_info` function.
```cpp
  autd->reads_fpga_info() = true;
  autd->update_ctrl_flag();
  const auto fpga_info = autd->fpga_info();
```
The return value of `fpga_info` is a `vector` of `uint8_t`, where the lowest $\SI{1}{bit}$ represents the fan state for each device.
All other bits are 0.

## Duty offset

To change $D_\text{offset}$ (see [Create Custom Gain Tutorial](gain.md#create-custom-gain-tutorial)), use the `set_duty_offset` function.
The argument of `set_duty_offset` is a `vector<array<uint8_t, 249>>`, which is set $D_\text{offset}$ for each transducer.
Note that only the lowest $\SI{1}{bit}$ is used, so only $D_\text{offset}=0,1$ can be used.

## Output delay

SDK provides a function to delay the output of each transducer relative to the unit of $\SI{25}{μs}$.
To do so, use the `set_output_delay` function.
The argument is a `vector<array<uint8_t, 249>>`.
Note that only the lower $\SI{7}{bit}$ of the delay value is used, and so the maximum delay is $127=\SI{3.175}{ms}$.

## pause/resume/stop

Call the `pause` function to pause the output.
It can be resumed with the `resume` function.

The `stop` function also stops the output, but cannot be resumed by `resume`.

Because the `pause` function stops the output abruptly (specifically, it takes the logical product of the output from the FPGA with 0), it may cause shutdown noise.
The `stop` function is designed to suppress it.

## clear

Clear the flags, `Gain`/`Modulation` data, etc. in the device.

## Firmware information

The `firmware_info_list` function can be used to get the version information of firmware.

```cpp
 for (auto&& firm_info : autd->firmware_info_list()) std::cout << firm_info << std::endl;
```

## Send functions

Send functions is a generic term for functions that actually send data to the device.
By calling these functions, the flags `output enable`, `silent mode`, `force fan`, `reads FPGA info`, and `output balance` are updated.
The behavior of these functions depends on the `check_ack` flag.
If `check_ack` is `true`, these functions will wait until the device actually processes the data, and return a bool value which represents whether the sending is succeeded.
Especially, when sending `Modulation`/`Sequence`, the processing time may increase significantly because the check is done every frame.
If `check_ack` is `false`, it does not check whether the data has been processed or not, and the return value is always `true`.

The following is a list of Send functions.

* `update_ctrl_flag`
* `set_output_delay`
* `set_duty_offset`
* `set_delay_offset`
* `clear`
* `close`
* `stop`
* `pause`
* `resume`
* `send`

[suzuki2020]: Suzuki, Shun, et al. "Reducing amplitude fluctuation by gradual phase shift in midair ultrasound haptics." IEEE transactions on haptics 13.1 (2020): 87-93.
