![build](https://github.com/shinolab/autd3-library-software/workflows/build/badge.svg)
![Upload Release Asset](https://github.com/shinolab/autd3-library-software/workflows/Upload%20Release%20Asset/badge.svg)

# [AUTD3](https://hapislab.org/airborne-ultrasound-tactile-display?lang=en)

Version: 1.8.2

## :blue_book: **[Wiki for beginners](https://github.com/shinolab/autd3-library-software/wiki)**

## :books: [API document](https://shinolab.github.io/autd3-library-software/index.html)

## :fire: CAUTION

* Before using, be sure to write the latest firmwares in `dist/firmware`. For more information, please see [README](/dist/firmware/README.md).

## :ballot_box_with_check: Requirements

* If you use `link::SOEM` on Windows, install [Npcap](https://nmap.org/npcap/) with WinPcap API-compatible mode (recommended) or [WinPcap](https://www.winpcap.org/).
    * This works on Windows 11

* If you use `link::TwinCAT` or `link::RemoteTwinCAT`, please see [how to install AUTDServer](https://github.com/shinolab/autd3-library-software/wiki/How-to-install-AUTDServer).
    * Windows 11 might not be supported by TwinCAT

## :hammer_and_wrench: Build

* Pre-built binaries and header files are on the [GitHub Release page](https://github.com/shinolab/autd3-library-software/releases). Instead, if you want to build from source, install CMake version 3.16 or higher and follow the instructions below.
    * Windows:
        ```
        git clone https://github.com/shinolab/autd3-library-software.git --recursive 
        ```
        Then, run `client/build.ps1` (Visual Studio 2017 or 2019 or 2022 is required) or build with CMake
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

## :nut_and_bolt: Firmware

* The firmware codes are available at [here](https://github.com/shinolab/autd3-library-firmware).

## :mortar_board: Citing

If you use this SDK in your research please consider to include the following citation in your publications:

* [S. Suzuki, S. Inoue, M. Fujiwara, Y. Makino and H. Shinoda, "AUTD3: Scalable Airborne Ultrasound Tactile Display," in IEEE Transactions on Haptics, doi: 10.1109/TOH.2021.3069976.](https://ieeexplore.ieee.org/document/9392322)
* S. Inoue, Y. Makino and H. Shinoda "Scalable Architecture for Airborne Ultrasound Tactile Display", Asia Haptics 2016

## :copyright: LICENSE

See [LICENSE](./LICENSE) and [NOTICE](./NOTICE).

# Author

Shun Suzuki, 2019-2021
