# Emulator

[autd-emulator](https://github.com/shinolab/autd-emulator) is a cross-platform emulator of AUTD3.

## Install

You can download the pre-compiled binary distributed on GitHub (https://github.com/shinolab/autd-emulator/releases) only for Windows 10 64bit.

For other OS, you have to install the rust compiler and compile it by yourself.
```
git clone https://github.com/shinolab/autd-emulator.git
cd autd-emulator
cargo run --release
```

## How to

<figure>
  <img src="https://raw.githubusercontent.com/shinolab/autd3-library-software/master/book/src/fig/Users_Manual/emu-home.jpg"/>
  <figcaption>Emulator</figcaption>
</figure>

When you launch autd-emulator, you will see the screen as shown above figure.
In this state, when you execute a client program using the Emulator link, the sound field corresponding to the content of the client program is displayed.
The black panel in the center of the figure is called "Slice", and you can visualize the sound field at any position by using this Slice.
The phase of the transducer is represented by the _hue_ and the amplitude by the _intensity_.

The sound field displayed by the emulator is a simple superposition of spherical waves, and it does not take into account the directivitiy and nonlinear effects.

The GUI on the left side of the screen can be used to control the slice and the camera.
The GUI is based on [Dear ImGui](https://github.com/ocornut/imgui), which can be operated by mouse or by "Ctrl+click" to enter values by a keyboard.

You can also move the camera by "dragging" and rotate the camera by "Shift+dragging".

### Slice tab

In the Slice tab, you can change the size, position, and rotation of the slice.
Rotation is specified by XYZ euler angles.
Clicking "xy", "yz", or "zx" button rotates the slice to the parallel state to each plane.

The intensity of the sound pressure is represented by color in Slice.
"Color scale" represents the maximum value of sound pressure in this color space.
If you use a large number of devices, the color may be saturated, in which case you should increase the value of "Color scale".
You can also specify the alpha value of Slice itself by `Slice alpha'.

Moreover, if you compile with `offscreen_renderer` feature enabled, you can save and record the sound field displayed on Slice[^1].

### Camera tab

In Camera tab, you can change camera position, rotation, field of view angle, near clip and far clip settings.
Rotation is specified by XYZ euler angle.

### Config tab

In the Config tab, you can set the wavelength, the alpha value of the transducers, and the background color.

In addition, after connecting to the Emulator link, you can switch the display/enable for each device.
When the display is turned off, the device is not displayed but contributes to the sound field.
If you turn off the enable, it does not contribute to the sound field.
In addition, the axis of each device can be displayed.

### Info tab

In the Info tab, you can check the information of Modulation and Sequence.

Modulation is not displayed on the Slice.
Instead, how the sound pressure is modulated is shown in this tab.
In raw mode, you can see how the duty ratio is modulated.

When a Sequence is sent, the information of the Sequence is displayed.
Sequence does not switch automatically, instead, you can use the "sequence index" to display the corresponding Sequence.

In the flag section, the Control flag is displayed.
Even if Silent mode is turned on, there is no change in the display on Slice.
Also, Output delay and Duty offset are ignored.

### Log tab

In the Log tab, you can see the log for debugging.

### Other settings

All settings are stored in `settings.json`.
Some settings can be edited only from `settings.json`.
The most important ones are "port" and "vsync".
"port" is the port number used to connect to the SDK's Emulator link.
If you set "vsync" to true, vertical synchronization is enabled.

[^1]: In the pre-built binary for Windows 10 64bit, it is turned on. Note that this function uses [Vulkano](https://github.com/vulkano-rs/vulkano), which is a wrapper for Vulkan, and you should check how to compile Vulkano when you compile it.
