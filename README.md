# autd3 #

Version: 3.0.2.3

* Old stable ver is [v3.0.0](https://github.com/shinolab/autd3-library-software/tree/v3.0.0)

* There is also [version 3.1-rc](https://github.com/shinolab/autd3.1) which is equipped with the a high-speed amp/phase switching feature up to 1.28MHz.

* For more details, refer to [Wiki](https://github.com/shinolab/autd3-library-software/wiki)

## CAUTION ##

Before using, be sure to write the v3.0.2 firmwares in `dist/firmware`

See [readme](/dist/firmware/readme)

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



## Wrappers ##

* C# - AUTDSharp
* Rust - future/autd
* python - wrappers/python
* julia - wrappers/julia

## FAQ/Troubleshooting

* "Dll not found exception"
    * if you installed Matlab, please add Matlab bin directory to PATH. (e.g. C:\Program Files\MATLAB\R2019b\bin\win64)


## Citing

If you use this SDK in your research please consider to include the following citation in your publications:

S. Inoue, Y. Makino and H. Shinoda "Scalable Architecture for Airborne Ultrasound Tactile Display", Asia Haptics 2016


# Author #

Shun Suzuki, 2019-2020
