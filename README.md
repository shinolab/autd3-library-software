![Windows](https://github.com/shinolab/autd3-library-software/workflows/Windows/badge.svg)
![Linux](https://github.com/shinolab/autd3-library-software/workflows/Linux/badge.svg)
![MacOS](https://github.com/shinolab/autd3-library-software/workflows/MacOS/badge.svg)
![Upload Release Asset](https://github.com/shinolab/autd3-library-software/workflows/Upload%20Release%20Asset/badge.svg?branch=v0.3)

# autd3 #

Version: 0.3.1

* This repository is forked from [old version](https://github.com/shinolab/autd)

* There is also [version 3.1-rc](https://github.com/shinolab/autd3.1) which is equipped with a high-speed amp/phase switching feature up to 1.28MHz.

* For more details, refer to [Wiki](https://github.com/shinolab/autd3-library-software/wiki)

## Versioning ##

The meanings of version number x.y.z are
* x: Architecture version.
* y: Firmware version.
* z: Software version.

If the number of x or y changes, the firmware of FPGA or CPU must be upgraded.

This versioning was introduced after version 0.3.0.

## CAUTION ##

Before using, be sure to write the v0.3 firmwares in `dist/firmware`

See [readme](/dist/firmware/Readme.md)

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

See `client/example_soem/simple_soem.cpp`

If you are using Linux/Mac, you may need to run as root.

## For other programming languages ##

* [Rust](https://github.com/shinolab/ruautd)
* [C#](https://github.com/shinolab/autd3sharp)
* [python](https://github.com/shinolab/pyautd)
* [julia](https://github.com/shinolab/AUTD3.jl)

## Citing

If you use this SDK in your research please consider to include the following citation in your publications:

S. Inoue, Y. Makino and H. Shinoda "Scalable Architecture for Airborne Ultrasound Tactile Display", Asia Haptics 2016


# Author #

Shun Suzuki, 2019-2020
