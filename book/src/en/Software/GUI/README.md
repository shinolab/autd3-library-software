# GUI

<figure>
  <img src="https://raw.githubusercontent.com/shinolab/autd3-library-software/master/book/src/fig/Users_Manual/GUI.jpg"/>
  <figcaption>GUI controller</figcaption>
</figure>

[AUTD3 GUI Controller](https://github.com/shinolab/AUTD3-GUI-Controller) is a program to operate AUTD with GUI.
It is made by using C\# and WPF, and currently, it works only on Windows.

## Install

There is a pre-compiled binary on GitHub, so please download it.

## How to

After starting the AUTD3 GUI Controller, the Home screen will be displayed as shown above.
You can configure various settings from the tabs on the left.

### Geometry

In Geometry tab, you can define Geometry.
Click "+" button to add a device and specify its position and rotation.
Currently, specifying rotation by quaternions is not supported.
Grouped Gain is not also supported, so you can't specify the group ID.
The order can be changed by drag & drop.

### Link

After defining the Geometry, select Link in the Link tab and click the Open button to connect.

### Gain & Modulation

After connecting the Link, select Gain/Modulation in the Gain/Modulation tab, and send it to the device with the "+" button at the bottom right.

Grouped Gain is not supported.

### Sequence

Gain Sequence is not supported, and only PointSequence can be used.

### Others

The File button at the top of the screen saves the current settings.
You can recall the saved settings by clicking the folder button on the right.
You can also use the green triangle button on the right to resume, and the red pause button on the right to pause.

To exit, press the upper right button.
When you exit, the current settings will be automatically saved and loaded at the next startup.
