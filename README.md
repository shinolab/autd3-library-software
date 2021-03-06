![build](https://github.com/shinolab/autd3-library-software/workflows/build/badge.svg)
![Upload Release Asset](https://github.com/shinolab/autd3-library-software/workflows/Upload%20Release%20Asset/badge.svg)

# [AUTD3](https://hapislab.org/airborne-ultrasound-tactile-display?lang=en)

Version: 0.9.0

* This repository is forked from [old version](https://github.com/shinolab/autd)

* [Here is API document](https://shinolab.github.io/autd3-library-software/index.html)

* For more details, refer to [Wiki](https://github.com/shinolab/autd3-library-software/wiki)

* The firmware codes are available at [here](https://github.com/shinolab/autd3-library-firmware).

## :memo: Versioning

The meanings of version number x.y.z are
* x: Architecture version.
* y: Firmware version.
* z: Software version.

If the number of x or y changes, the firmware of FPGA or CPU must be upgraded.

This versioning was introduced after version 0.3.0.

## :fire: CAUTION

* Before using, be sure to write the latest firmwares in `dist/firmware`. For more information, please see [readme](/dist/firmware/Readme.md).

## :ballot_box_with_check: Requirements

* If you use `SOEMLink` on Windows, install [Npcap](https://nmap.org/npcap/) with WinPcap API-compatible mode (recommended) or [WinPcap](https://www.winpcap.org/).

* If you use `TwinCAT`, please see [how to install AUTDServer](https://github.com/shinolab/autd3-library-software/wiki/How-to-install-AUTDServer).

## :hammer_and_wrench: Build

* Pre-built binaries and header files are on the [GitHub Release page](https://github.com/shinolab/autd3-library-software/releases). Instead, if you want to build from source, install CMake version 3.12 or higher and follow the instructions below.
    * Windows:
        ```
        git clone https://github.com/shinolab/autd3-library-software.git --recursive 
        ```
        Then, run `client/build.ps1` (Visual Studio 2019 is required) or build with CMake
    * Linux/macOS: 
        ```
        git clone https://github.com/shinolab/autd3-library-software.git --recursive
        cd autd3-library-software/client
        mkdir build && cd build
        cmake ..
        make
        ```

    * Some projects are disabled by default. Please enable them by switching their flags ON
        * e.g., if you want to use TwinCATLink;
            ```
            cmake .. -DBUILD_TWINCAT_LINK=ON
            ```

    * See [README](./client/README.md) for more details. 

## :beginner: Example

See `client/examples`

If you are using Linux/macOS, you may need to run as root.

## :link: For other programming languages

* [Rust](https://github.com/shinolab/rust-autd)
* [C#](https://github.com/shinolab/autd3sharp)
* [python](https://github.com/shinolab/pyautd)
* [julia](https://github.com/shinolab/AUTD3.jl)

## :mortar_board: Citing

If you use this SDK in your research please consider to include the following citation in your publications:

* S. Inoue, Y. Makino and H. Shinoda "Scalable Architecture for Airborne Ultrasound Tactile Display", Asia Haptics 2016

## :copyright: LICENSE

See [LICENSE](./LICENSE) and [NOTICE](./NOTICE).

# Author

Shun Suzuki, 2019-2021
