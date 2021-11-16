# Concept

The primary classes that make up the SDK are as follows,

* `Controller` - All operations for AUTD3 are performed via this class
* `Geometry` - Manage the geometry of devices in the real world
* `Link` - Interface to the devices
* `Gain` - Class to manage the phase/amplitude of each transducer
* `Modulation` - Class to manage Amplitude Modulation (AM)
* `Sequence` - Class to manage Spatio-Temporal Modulation (STM) functions on Hardware

The flow of using SDK is as follows,

* Create a `Controller`
* Setting the position and orientation of connected devices
* Create `Link`, and connect to devices 
* Create and send `Gain`/`Sequence`, and/or `Modulation`

Here is a picture of the AUTD3 from the top.

<figure>
  <img src="https://raw.githubusercontent.com/shinolab/autd3-library-software/master/book/src/fig/Users_Manual/autd_trans_idx.jpg"/>
  <figcaption>AUTD front</figcaption>
</figure>

The image of the back of AUTD3 is shown below. 
The connector for the 24V power supply is _Molex 5566-02A_.

<figure>
  <img src="https://raw.githubusercontent.com/shinolab/autd3-library-software/master/book/src/fig/Users_Manual/autd_back.jpg"/>
  <figcaption>AUTD back</figcaption>
</figure>

Each unit of AUTD3 consists of 249 transduces[^fn_asm], each of which is assigned an index number as shown in the above figure.
From SDK, the phase/amplitude of all these transducers can be controlled individually in $\SI{8}{bit}$ resolutions.

The coordinate system of AUTD3 is a right-handed, where the center of the 0-th transducer is the origin.
The $x$-axis is in the direction of the major axis, i.e., 0→17, and the $y$-axis is in the direction 0→18.
The unit system in SDK is $\SI{}{mm}$ for distance, $\SI{}{rad}$ for angle, and $\SI{}{Hz}$ for frequency.
The transducers are arranged at intervals of $\SI{10.16}{mm}$, and the total size including the substrate is $\SI{192}{mm}\times\SI{151.4}{mm}$.
The outline drawing of the transducers array is shown below.

<figure>
  <img src="https://raw.githubusercontent.com/shinolab/autd3-library-software/master/book/src/fig/Users_Manual/transducers_array.jpg"/>
  <figcaption>Design drawing of transducer array</figcaption>
</figure>

In addition, multiple AUTD3 units can be connected and extended by daisy-chaining.
An extended array can be configured by connecting a PC to the first `EherCAT In` via an ethernet cable and connecting the $i$-th `EherCAT Out` to the $i+1$-th `EherCAT In`.
The ethernet cable used must be CAT 5e or higher category.

[^fn_asm]: Three transducers are missing from $18\times 14=252$. The reason why the screw holes are placed at their positions is to minimize the gaps when multiple units are placed densely.
