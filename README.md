![Windows](https://github.com/shinolab/autd3-library-software/workflows/Windows/badge.svg?branch=v0.3)
![Linux](https://github.com/shinolab/autd3-library-software/workflows/Linux/badge.svg?branch=v0.3)
![MacOS](https://github.com/shinolab/autd3-library-software/workflows/MacOS/badge.svg?branch=v0.3)
[![Upload Release Asset](https://github.com/shinolab/autd3-library-software/workflows/Upload%20Release%20Asset/badge.svg?branch=v0.3)](https://github.com/shinolab/autd3-library-software/releases)

# autd3 #

Version: 3.0.3.0

* Old stable ver is [v3.0.0](https://github.com/shinolab/autd3-library-software/tree/v3.0.0)

* There is also [version 3.1-rc](https://github.com/shinolab/autd3.1) which is equipped with the a high-speed amp/phase switching feature up to 1.28MHz.

* For more details, refer to [Wiki](https://github.com/shinolab/autd3-library-software/wiki)

## Versioning ##

The meanings of version number 3.x.y.z are
* x: Architecture version.
* y: Firmware version. 
* z: Software version.

If the number of x or y changes, the firmware of FPGA or CPU must be upgraded.

This versioning was introduced after version 3.0.3.0.

## CAUTION ##

Before using, be sure to write the v3.0.3 firmwares in `dist/firmware`

See [readme](/dist/firmware/readme.md)

## Requirements

* If you are using Windows, install [Npcap](https://nmap.org/npcap/) with WinPcap API-compatible mode (recomennded) or [WinPcap](https://www.winpcap.org/).

## Build ##

* Windows: run `client/build.ps1`
* Linux/Mac: 
    ```
        cd client
        mkdir build && cd build
        cmake ..
        make
    ```

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
