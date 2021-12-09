# README

## CMake options

[] is a default.

* BUILD_ALL = [OFF]
* BUILD_DOC = [OFF]
* BUILD_HOLO_GAIN = [ON]
* BUILD_BLAS_BACKEND = [OFF]
* USE_MKL = [OFF]
* BUILD_CUDA_BACKEND = [OFF]
* BUILD_ARRAYFIRE_BACKEND = [OFF]
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
* -VS_VERSION = 2017, 2019, [2022]
* -ARCH = [x64]

## Linux/macOS

```
mkdir build && cd build
cmake ..
make
sudo ./examples/example_soem
```

# Build BLAS backend for HoloGain

```
cmake .. -DBUILD_HOLO_GAIN=ON -DBUILD_BLAS_BACKEND=ON -DBLAS_LIB_DIR=<your BLAS library path> -DBLAS_INCLUDE_DIR=<your BLAS include path> -DBLA_VENDOR=<your BLAS vendor>
```

* If you use Intel MKL, please set `USE_MKL` ON.
    ```
    cmake .. -DBUILD_HOLO_GAIN=ON -DBUILD_BLAS_BACKEND=ON -DBLAS_LIB_DIR=<your MKL library path> -DBLAS_INCLUDE_DIR=<your MKL include path> -DBLA_VENDOR=Intel10_64lp -DUSE_MKL=ON
    ```

## OpenBLAS install example in Windows

* Following is an example to install [OpenBLAS](https://github.com/xianyi/OpenBLAS). See also [official instruction](https://github.com/xianyi/OpenBLAS/wiki/How-to-use-OpenBLAS-in-Microsoft-Visual-Studio).
    * Install Visual Studio 2022 and Anaconda (or miniconda), then open Anaconda Prompt.
        ```
        git clone https://github.com/xianyi/OpenBLAS
        cd OpenBLAS
        conda update -n base conda
        conda config --add channels conda-forge
        conda install -y cmake flang clangdev perl libflang ninja
        "c:/Program Files/Microsoft Visual Studio/2022/Community/VC/Auxiliary/Build/vcvars64.bat"
        set "LIB=%CONDA_PREFIX%\Library\lib;%LIB%"
        set "CPATH=%CONDA_PREFIX%\Library\include;%CPATH%"
        mkdir build
        cd build
        cmake .. -G "Ninja" -DCMAKE_CXX_COMPILER=clang-cl -DCMAKE_C_COMPILER=clang-cl -DCMAKE_Fortran_COMPILER=flang -DCMAKE_MT=mt -DBUILD_WITHOUT_LAPACK=no -DNOFORTRAN=0 -DDYNAMIC_ARCH=ON -DCMAKE_BUILD_TYPE=Release
        cmake --build . --config Release
        cmake --install . --prefix D:\lib\openblas -v
        ```
    * You can set install prefix path anywhere you want
    * Also, you may need to add `%CONDA_HOME%\Library\bin` to PATH, where `CONDA_HOME` is a home directory path of Anaconda (or miniconda).

* Then, compile 
    ```
    cmake .. -DBUILD_HOLO_GAIN=ON -DBUILD_BLAS_BACKEND=ON -DBLAS_LIB_DIR=D:/lib/openblas -DBLAS_INCLUDE_DIR=D:/lib/openblas/include/openblas -DBLA_VENDOR=OpenBLAS
    ```

    * If you get `flangxxx.lib` link error, add `-DBLAS_DEPEND_LIB_DIR=<your conda path>/Library/lib` option

* Tested OpenBLAS version is 0.3.18

# Build CUDA backend for HoloGain

* Install CUDA Toolkit, and set `BUILD_CUDA_BACKEND` ON
  * Tested CUDA Toolkit version is 11.5.50

```
cmake .. -DBUILD_HOLO_GAIN=ON -DBUILD_CUDA_BACKEND=ON
```

# Build ArrayFire backend for HoloGain

* Install ArrayFire, and set `BUILD_ARRAYFIRE_BACKEND` ON
    * You may need to add `%AF_PATH%\lib` to PATH, where `AF_PATH` is your ArrayFire install directory

```
cmake .. -DBUILD_HOLO_GAIN=ON -DBUILD_ARRAYFIRE_BACKEND=ON
```

# Author

Shun Suzuki, 2019-2021
