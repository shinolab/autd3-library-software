# README

## CMake options

[] is a default.

* -BUILD_ALL = [OFF]
* -BUILD_DOC = [OFF]
* -USE_DOUBLE = [OFF]
* -DISABLE_EIGEN = [OFF]
* -ENABLE_BLAS = [OFF]
* -BLAS_LIB_DIR = []
* -BLAS_INCLUDE_DIR = []
* -BLAS_DEPEND_LIB_DIR = []
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

# Use BLAS

BLAS can be used in building HoloGain (multiple foci), however, you have to install BLAS independently.

* **Only OpenBLAS and Intel MKL are tested**; however, other BLAS libraries also may work. 

## Build with BLAS

```
mkdir build
cd build
cmake .. -DENABLE_BLAS=ON
```
or
```
cmake .. -DENABLE_BLAS=ON -DBLAS_LIB_DIR=<your BLAS library path> -DBLAS_INCLUDE_DIR=<your BLAS include path>
```

* If you are using Windows, you may need to set `BLAS_DEPEND_LIB_DIR` to link some additional libraries.
    * For example, if you installed OpenBLAS as follow the below install example, you need link `flangmain.lib` by the following command;
        ```
        cmake .. -DENABLE_BLAS=ON -DBLAS_LIB_DIR=C:/opt/lib -DBLAS_INCLUDE_DIR=C:/opt/include/openblas -DBLAS_DEPEND_LIB_DIR=<your conda path>/Library/lib
        ``` 

* If you use Intel MKL, please set `USE_MKL` ON.
    ```
    cmake .. -DENABLE_BLAS=ON -DBLAS_LIB_DIR=<your MKL lib path> -DBLAS_INCLUDE_DIR=<your MKL include path> -DUSE_MKL=ON
    ```

## OpenBLAS install example in Windows

* Following is an example to install [OpenBLAS](https://github.com/xianyi/OpenBLAS). See also [official instruction](https://github.com/xianyi/OpenBLAS/wiki/How-to-use-OpenBLAS-in-Microsoft-Visual-Studio).
    * Install Visual Studio 2019 and Anaconda, then open Anaconda Prompt.
        ```
        git clone https://github.com/xianyi/OpenBLAS
        cd OpenBLAS
        conda update -n base conda
        conda config --add channels conda-forge
        conda install -y cmake flang clangdev perl libflang ninja
        "c:/Program Files (x86)/Microsoft Visual Studio/2019/Community/VC/Auxiliary/Build/vcvars64.bat"
        set "LIB=%CONDA_PREFIX%\Library\lib;%LIB%"
        set "CPATH=%CONDA_PREFIX%\Library\include;%CPATH%"
        mkdir build
        cd build
        cmake .. -G "Ninja" -DCMAKE_CXX_COMPILER=clang-cl -DCMAKE_C_COMPILER=clang-cl -DCMAKE_Fortran_COMPILER=flang -DBUILD_WITHOUT_LAPACK=no -DNOFORTRAN=0 -DDYNAMIC_ARCH=ON -DCMAKE_BUILD_TYPE=Release
        cmake --build . --config Release
        cmake --install . --prefix c:\opt -v
        ```

# Author

Shun Suzuki, 2019-2020
