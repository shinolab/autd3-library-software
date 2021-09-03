# README

## CMake options

[] is a default.

* BUILD_ALL = [OFF]
* BUILD_DOC = [OFF]
* BUILD_HOLO_GAIN = [ON]
* BUILD_FROM_FILE_MODULATION = [OFF]
* BUILD_SOEM_LINK = [ON]
* BUILD_TWINCAT_LINK = [OFF]
* BUILD_EMULATOR_LINK = [OFF]
* BUILD_EXAMPLES = [ON]
* BUILD_CAPI = [OFF]
* ENABLE_LINT = [OFF]
* BUILD_TEST = [OFF]

## Windows

run `build.ps1` or run CMake, then open `autd3.sln` in `BUILD_DIR` (default `./build`)

### build.ps1 options

[] is a default.

* -BUILD_DIR = [./build]
* -VS_VERSION = 2017, [2019], 2022
* -ARCH = [x64]

## Linux/macOS

```
mkdir build && cd build
cmake ..
make
sudo ./examples/example_soem
```

# Author

Shun Suzuki, 2019-2021
