# README #

## Windows ##

run `build.ps1`, then open `autd3.sln` in `BUILD_DIR` (default `build`)

### Option ###

[] is a default.

* -BUILD_DIR = [\build]
* -VS_VERSION = 2017, [2019]
* -ARCH = [x64]
* -DISABLE_MATLAB = [False]
* -USE_DOUBLE = [False]

## Mac/Linux ##

```
mkdir build && cd build
cmake ..
make
sudo examples/example_soem
```

# Author #

Shun Suzuki, 2019-2020
