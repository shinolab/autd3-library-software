# README #

## CMake options

[] is a default.

* -BUILD_ALL = [OFF]
* -BUILD_DOC = [OFF]
* -USE_DOUBLE = [OFF]
* -ENABLE_EIGEN = [ON]
* -BUILD_HOLO_GAIN = [ON]
* -BUILD_MATLAB_GAIN = [OFF]
* -BUILD_FROM_FILE_MODULATION = [OFF]
* -BUILD_SOEM_LINK = [ON]
* -BUILD_TWINCAT_LINK = [OFF]
* -BUILD_DEBUG_LINK = [OFF]
* -BUILD_EMULATOR_LINK = [OFF]
* -IGNORE_EXAMPLE = [OFF]
* -BUILD_CAPI = [OFF]
* -ENABLE_LINT = [OFF]

## Windows

run `build.ps1` or run CMake, then open `autd3.sln` in `BUILD_DIR` (default `./build`)

### build.ps1 options

[] is a default.

* -BUILD_DIR = [./build]
* -VS_VERSION = 2017, [2019]
* -ARCH = [x64]
* -BUILD_ALL = [False]
* -USE_DOUBLE = [False]

## Mac/Linux

```
mkdir build && cd build
cmake ..
make
sudo examples/example_soem
```

# Author

Shun Suzuki, 2019-2020
