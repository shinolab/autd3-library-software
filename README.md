![build](https://github.com/shinolab/autd3-library-software/workflows/build/badge.svg)
![Upload Release Asset](https://github.com/shinolab/autd3-library-software/workflows/Upload%20Release%20Asset/badge.svg)

# autd3 #

Version: 0.4.1

* This repository is forked from [old version](https://github.com/shinolab/autd)

* [Here is API document](https://shinolab.github.io/autd3-library-software/index.html)

* For more details, refer to [Wiki](https://github.com/shinolab/autd3-library-software/wiki)

## Versioning ##

The meanings of version number x.y.z are
* x: Architecture version.
* y: Firmware version.
* z: Software version.

If the number of x or y changes, the firmware of FPGA or CPU must be upgraded.

This versioning was introduced after version 0.3.0.

## ⚠ CAUTION ##

* Before using, be sure to write the v0.4 firmwares in `dist/firmware`. For more information, please see [readme](/dist/firmware/Readme.md).

* If you are using Windows, you should disable Hyper-V, otherwise, it will cause unexpected behavior.

## Requirements

* If you are using Windows, install [Npcap](https://nmap.org/npcap/) with WinPcap API-compatible mode (recommended) or [WinPcap](https://www.winpcap.org/).

## Build ##

* Pre-built binaries and header files are on the [GitHub Release page](https://github.com/shinolab/autd3-library-software/releases). Instead, if you want to build from source, install CMake version 3.12 or higher and follow the instructions below.
    * Windows:
        ```
        git clone https://github.com/shinolab/autd3-library-software.git --recursive 
        ```
        Then,  run `client/build.ps1` (Visual Studio 2019 is required)
    * Linux/Mac: 
        ```
        git clone https://github.com/shinolab/autd3-library-software.git --recursive
        cd autd3-library-software/client
        mkdir build && cd build
        cmake ..
        make
        ```

## Example

See `client/examples`

If you are using Linux/Mac, you may need to run as root.

## For other programming languages ##

* [Rust](https://github.com/shinolab/ruautd)
* [C#](https://github.com/shinolab/autd3sharp)
* [python](https://github.com/shinolab/pyautd)
* [julia](https://github.com/shinolab/AUTD3.jl)

## Citing

If you use this SDK in your research please consider to include the following citation in your publications:

S. Inoue, Y. Makino and H. Shinoda "Scalable Architecture for Airborne Ultrasound Tactile Display", Asia Haptics 2016

## Version 1.0.0-rc

* There is also [version 1.0.0-rc](https://github.com/shinolab/autd3.1) which is equipped with a high-speed amp/phase switching feature up to 1.28MHz.

# Author #

Shun Suzuki, 2019-2020
