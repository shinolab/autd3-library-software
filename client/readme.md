# README #

## Windows ##

run `build.ps1`

### Option ###

[] is a default.

* -BUILD_DIR = [\build]
* -VS_VERSION = 2017, [2019]
* -ARCH = [x64]
* -DISABLE_MATLAB = [False]
* -ENABLE_TEST = [False]
* -TOOL_CHAIN = [""]

### Caution: Unit test ###

Using vcpkg, please install gtest with vcpkg.

* PowerShell
    ```
    -ENABLE_TEST -TOOL_CHAIN "-DCMAKE_TOOLCHAIN_FILE=C:[...]\vcpkg\scripts\buildsystems\vcpkg.cmake"
    ```

* CMD
    ```
    -test "-DCMAKE_TOOLCHAIN_FILE=C:[...]\vcpkg\scripts\buildsystems\vcpkg.cmake"
    ```

## Mac/Linux ##

```
mkdir build && cd build
cmake ..
make
sudo exmaple_soem/simple_soem
```

# Author #

Shun Suzuki, 2019-2020