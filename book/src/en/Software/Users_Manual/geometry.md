# Geometry

In this section, we describe Geometry, which manages how AUTD3 devices are placed in the real world.

## Multiple devices

AUTD3 devices can be daisy-chained to each other.
To connect multiple devices, connect the PC to the first `EtherCAT In ` with a ethernet cable, and connect the $i$-th `EtherCAT Out` to the $i+1$-th `EtherCAT In` with a ethernet cables (see [Concept](concept.md)).
The power supply can also be daisy-chained, and you can use any of the three power supply connectors.

If you want to use multiple devices in the SDK, call the `add_device` function for each connected device.
The first argument of the `add_device` function is the position and the second argument is the rotation.
The rotation is specified by ZYZ euler angles or Quaternion.

<figure>
  <img src="https://raw.githubusercontent.com/shinolab/autd3-library-software/master/book/src/fig/Users_Manual/autd_hori.jpg"/>
  <figcaption>Horizontal alignment</figcaption>
</figure>

For example, suppose that the devices are arranged and connected as shown in the figure above, and the device on the left is the first device that is connected to the PC and the device on the right is the second device.
Assuming that the first device is placed at the global origin, you should call `add_device` as follows,
```cpp
  autd.geometry().add_device(autd::Vector3(0, 0, 0), autd::Vector3(0, 0, 0));
  autd.geometry().add_device(autd::Vector3(autd::DEVICE_WIDTH, 0, 0), autd::Vector3(0, 0, 0));
```
Here, `autd::DEVICE_WIDTH` is the width of the device (including the circuit board outline).
Since there is no rotation, the second argument should be `(0, 0, 0)`.

Or, for example, if you place the second device at the global origin, you should,
```cpp
  autd.geometry().add_device(autd::Vector3(-autd::DEVICE_WIDTH, 0, 0), autd::Vector3(0, 0, 0));
  autd.geometry().add_device(autd::Vector3(0, 0, 0), autd::Vector3(0, 0, 0));
```

<figure>
  <img src="https://raw.githubusercontent.com/shinolab/autd3-library-software/master/book/src/fig/Users_Manual/autd_vert.jpg"/>
  <figcaption>Vertical alignment</figcaption>
</figure>

For another example, if two devices are placed as shown in the figure above, with the first one on the bottom and the second one on the left, and the first device is placed at the global origin, you should specify the rotation as follows,
```cpp
  autd.geometry().add_device(autd::Vector3(0, 0, 0), autd::Vector3(0, 0, 0));
  autd.geometry().add_device(autd::Vector3(0, 0, autd::DEVICE_WIDTH), autd::Vector3(0, M_PI / 2.0, 0));
```

The global coordinate system is used in all SDK APIs, so you can use APIs transparently even when multiple devices are connected.
