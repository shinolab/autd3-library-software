# Getting Started

This section will describe how to use AUTD3 in practice.
In the following instructions, Windows 10 64bit will be used, and please modify the instructions appropriately if you use other OS.

## Install 

First, install build tools.
The tools and versions used in this section are as follows.
Please follow the official instructions to install each of them.
For Visual Studio Community, install "Desktop Development with C++".
If you are using Linux, you can use gcc. If you are using macOS, you can use clang.
In addition, since the following instruction will be operated from a terminal, you should set PATH appropriately.

* Visual Studio Community 2022 17.0.4
* CMake 3.22.1
* git 2.34.1.windows.1
* npcap 1.60

## Setup Device

Next, let's set up the devices.
Here we will use only one AUTD3 device.
Please connect the ethernet port of the PC to the `EtherCAT In` of the AUTD3 device (see [Concept](concept.md)) with an ethernet cable.
Next, connect the $\SI{24}{V}$ power supply.

### Firmware update

If the firmware is old, the operation is not guaranteed.
The version of firmware in this document is assumed to be 1.9.

To update the firmware, you need a Windows 10 64bit PC with [Vivado Design Suite](https://www.xilinx.com/products/design-tools/vivado.html) and [J-Link Software](https://www.segger.com/downloads/jlink/) installed.
We have confirmed that the update script works with Vivado 2021.1 and J-Link Software v7.58b (x64).

First, connect the AUTD3 device and the PC via [XILINX Platform Cable](https://www.xilinx.com/products/boards-and-kits/hw-usb-ii-g.html) and [J-Link Plus](https://www.segger.com/products/debug-probes/j-link/models/j-link-plus/) with [J-Link 9-Pin Cortex-M Adapter](https://www.segger-pocjapan.com/j-link-9-pin-cortex-m-adapter), and turn on the AUTD3.
Then, run `dist/firmware/autd_firmware_writer.ps1` in [SDK](https://github.com/shinolab/autd3-library-software).
The update will take a few minutes.

## Building first program

First, open a terminal and prepare an directory.
```
  mkdir autd3_sample
  cd autd3_sample
```

Next, make `CMakeLists.txt` and `main.cpp` files.
```
└─autd3_sample
        CMakeLists.txt
        main.cpp
```

Next, download the latest binary version of the SDK.
The binaries are available at [GitHub Release](https://github.com/shinolab/autd3-library-software/releases).
Unzip the downloaded binary, and copy the `include` and `lib` folders to the `autd3_sample` folder.
```
└─autd3_sample
    │  CMakeLists.txt
    │  main.cpp
    ├─include
    └─lib
```

Next, download Eigen3, which is a header-only library for matrix computation.
Here, we change the current directory to `autd3_sample` and add Eigen3 as git submodule.
```
  git init
  git submodule add https://gitlab.com/libeigen/eigen.git eigen
  cd eigen
  git checkout 3.4.0
  cd ..
```
Alternatively, you can directly download [Eigen3](https://gitlab.com/libeigen/eigen) and put it under the `autd3_sample` folder. The Eigen3 version used in the SDK is 3.4.0.

At this time, the directory structure is as follows.
```
└─autd3_sample
    │  CMakeLists.txt
    │  main.cpp
    ├─include
    ├─lib
    └─eigen
        ├─bench
        ├─blas
        ├─ci
        ├─cmake
        ...
```

Next, write `CMakeLists.txt` as follows.
```
cmake_minimum_required(VERSION 3.16)

project(autd3_sample)
set (CMAKE_CXX_STANDARD 17)

add_executable(main main.cpp)

target_compile_definitions(main PRIVATE _USE_MATH_DEFINES)
target_link_directories(main PRIVATE lib)
target_include_directories(main PRIVATE include eigen)

if(WIN32)
  target_link_directories(main PRIVATE lib/wpcap)
  target_link_libraries(main autd3 soem_link Packet.lib wpcap.lib ws2_32.lib winmm.lib)
elseif(APPLE)
  target_link_libraries(main pcap)
else()
  target_link_libraries(main rt)
endif()

if(WIN32)
    set_property(DIRECTORY \${CMAKE_CURRENT_SOURCE_DIR} PROPERTY VS_STARTUP_PROJECT main)
endif()
```

And, write `main.cpp` as follows.
This source code will apply $\SI{150}{Hz}$ AM modulation to a single focus at $\SI{150}{mm}$ above of center of the array.
```cpp
#include <iostream>

#include "autd3.hpp"
#include "autd3/link/soem.hpp"

using namespace std;
using namespace autd;

string get_adapter_name() {
  size_t i = 0;
  const auto adapters = link::SOEM::enumerate_adapters();
  for (auto&& [desc, name] : adapters) cout << "[" << i++ << "]: " << desc << ", " << name << endl;

  cout << "Choose number: ";
  string in;
  getline(cin, in);
  stringstream s(in);
  if (const auto empty = in == "\n"; !(s >> i) || i >= adapters.size() || empty) return "";

  return adapters[i].name;
}

int main() try {
  autd::Controller autd;

  autd.geometry().add_device(Vector3(0, 0, 0), Vector3(0, 0, 0));

  const auto ifname = get_adapter_name();
  auto link = link::SOEM::create(ifname, autd.geometry().num_devices());
  autd.open(std::move(link));

  autd.clear();

  auto firm_info_list = autd.firmware_info_list();
  for (auto&& firm_info : firm_info_list) cout << firm_info << endl;

  autd.silent_mode() = true;

  const auto focus = Vector3(TRANS_SPACING_MM * ((NUM_TRANS_X - 1) / 2.0), TRANS_SPACING_MM * ((NUM_TRANS_Y - 1) / 2.0), 150.0);
  gain::FocalPoint g(focus);
  modulation::Sine m(150);
  autd << g, m;

  cout << "press enter to finish..." << endl;
  cin.ignore();

  autd.close();

  return 0;
} catch (exception& ex) {
  cerr << ex.what() << endl;
}
```

Then, build with CMake.
```
  mkdir build
  cd build
  cmake .. -G "Visual Studio 17 2022" -A x64
```
`autd3_sample.sln` should be generated under the _build_ directory, so open it and run the _main_ project.
**Note that you should change the build configuration of Visual Studio from Debug to Release**.
In case of Linux/macOS, you may need to run the program as root user.

## Explanation

Here, we will explain the above codes.

To use the SDK, include the `autd3.hpp` header.
You also need `autd3/link/soem.hpp` to use `link::SOEM`.
```cpp
#include "autd3.hpp"
#include "autd3/link/soem.hpp"
```

Then, create `Controller` instance.
```cpp
  autd::Controller autd;
```

After that, we specify the geometry of the device in the real world.
```cpp
  autd.geometry().add_device(autd::Vector3(0, 0, 0), autd::Vector3(0, 0, 0));
```
The first argument of `add_device` is the position, the second argument is the rotation.
The position is 0-th transducer position in the global coordinate system you set.
The rotation is specified by ZYZ euler angles or quaternions.
Here, we assume that the device is placed at global origin without rotation.

Next, create `Link`, and connect to the device.
```cpp
  const auto ifname = get_adapter_name();
  auto link = link::SOEM::create(ifname, autd.geometry().num_devices());
  autd.open(std::move(link));
```
The first argument of `link::SOEM::create()` is the ethernet interface name where the AUTD3 device is connected, and the second argument is the number of AUTD3 devices connected. 
We prepared a utility function `get_adapter_name` to get the interface name list, so please select the appropriate one at runtime.
(On macOS or Linux, you can also use `ifconfig` to check it.)

Then, initialize the AUTD devices.
You may not need to call `clear()` since it is initialized at power-on.
```cpp
  autd.clear();
```

Next, we check the version of the firmware.
This operation is not required to run AUTD3.
```cpp
  auto firm_info_list = autd.firmware_info_list();
  for (auto&& firm_info : firm_info_list) cout << firm_info << endl;
```

Next, we set _silent mode_ on.
```cpp
  autd.silent_mode() = true;
```
Since it is on by default, you don't need to call it in fact.
If you want to turn it off, please give `false`.
In _silent mode_, the phase/amplitude parameters given to the transducer are passed through a low-pass filter to reduce noise.

After that, send the `Modulation` which applies $\SI{150}{Hz}$ Sin wave amplitude modulation and the `Gain` which represents the single focus.
```cpp
  const auto point = autd::Vector3(autd::TRANS_SPACING_MM * ((autd::NUM_TRANS_X - 1) / 2.0), autd::TRANS_SPACING_MM * ((autd::NUM_TRANS_Y - 1) / 2.0), 150.0);
  autd::gain::FocalPoint g(point);
  autd::modulation::Sine m(150);
  autd << g, m;
```
The `point` is a bit complicated; `TRANS_SPACING_MM` represents the spacing of the transducers, and `NUM_TRANS_X` and `NUM_TRANS_Y` represent the number of transducers in the $x,y$-axis, respectively.
Therefore, `point` represents the point $\SI{150}{mm}$ right above the center of the transducers array.

Finally, you should disconnect the device.
```cpp
  autd.close();
```

In the next section, we will describe the basic functions.
