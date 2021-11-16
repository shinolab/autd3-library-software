# Gain

This SDK can control the phase/amplitude of each transducer individually, which allows to generate various sound fields.
`Gain` is a class to manage this, and the SDK provides `Gain`s for generating several kinds of sound fields by default.

## FocalPoint

`FocalPoint` is the simplest `Gain`, which produces a single focus.
```cpp
    const auto g = autd::gain::FocalPoint::create(autd::Vector3(x, y, z));
```
The first argument of `FocalPoint::create` is the position of the focus.
As the second argument, you can specify the amplitude as a duty ratio (`uint8_t`) or a normalized sound pressure amplitude from 0 to 1 (`double`).

Here we note the relationship between the duty ratio $D$ and the sound pressure $p$.
Theoretically,
$$
    p \propto \sin \pi D
$$
Therefore, the sound pressure takes the minimum value $p=0$ at duty ratio $D=0$ and the maximum value at duty ratio $D=\SI{50}{\%}$, but the relationship between them is not linear.
In the case of specifying the sound pressure amplitude, $p=1$ is taken as the maximum value, and it is internally converted to duty ratio by the inverse conversion of the above equation.

## BesselBeam

`BesselBeam` Gain generates a Bessel beam.
This `Gain` is based on the paper by Hasegawa et al.[hasegawa2017].
```cpp
  const autd::Vector3 apex(x, y, z);
  const autd::Vector3 dir = autd::Vector3::UnitZ();
  const double theta_z = 0.3;
  const auto g = autd::gain::BesselBeam::create(apex, dir, theta_z);
```

The first argument is the apex of the virtual cone that generates the beam, the second argument is the direction of the beam, and the third argument is the angle between the plane perpendicular to the beam and the side of the virtual cone ($\theta_z$ in the figure below).
As the fourth argument, the amplitude can be specified as a duty ratio (`uint8_t`) or a normalized sound pressure amplitude of 0-1 (`double`).

<figure>
  <img src="https://raw.githubusercontent.com/shinolab/autd3-library-software/master/book/src/fig/Users_Manual/1.4985159.figures.online.f1.jpg"/>
  <figcaption>Bessel beam (cited from [hasegawa2017])</figcaption>
</figure>

## PlaneWave

`PlaneWave` Gain generates plane wave.
```cpp
    const auto g = autd::gain::PlaneWave::create(autd::Vector3(x, y, z));
```
The first argument of `PlaneWave::create` is the direction of the plane wave.
As the second argument, you can specify the amplitude as a duty ratio (`uint8_t`) or a normalized sound pressure amplitude from 0 to 1 (`double`).

## TransducerTest

`TransducerTest` Gain drives a single transducer for debugging.
```cpp
    const auto g = autd::gain::TransducerTest::create(index, duty, phase);
```
The first argument of `TransducerTest::create` is the index of the transducer, the second argument is the duty ratio, and the third argument is the phase.

## Null

`Null` Gain is `Gain` with zero amplitudes, so it produces nothing.
```cpp
    const auto g = autd::gain::Null::create();
```

## Holo (Multiple foci)

`Holo` is a `Gain` for generating multi-foci.
Several algorithms have been proposed to generate multi-foci, and the following algorithms are implemented in the SDK.

* `SDP` - Semidefinite programming, based on [inoue2015]
* `EVD` - Eigen value decomposition, based on [long2014]
* `Naive` - Linear synthesis of single-focus solutions
* `GS` - Gershberg-Saxon, based on [marzo2019]
* `GSPAT` - Gershberg-Saxon for Phased Arrays of Transducers, based on [plasencia2020]
* `LM` - Levenberg-Marquardt, based on [levenberg1944, marquardt1963, madsen2004]
* `GaussNewton` - Gauss-Newton
* `GradientDescent` - Gradient descent
* `APO` - Acoustic Power Optimization, based on [hasegawa2020]
* `Greedy` - Greedy algorithm and Brute-force search, based on [suzuki2021]

In addition, each method has a choice of computational backend.
The following `Backend` is provided in the SDK.

* `EigenBackend` - uses [Eigen](https://eigen.tuxfamily.org/index.php?title=Main_Page). This backend is enabled by default.
* `BLASBackend` - uses BLAS/LAPACK such as [OpenBLAS](https://www.openblas.net/) and [Intel MKL](https://software.intel.com/content/www/us/en/develop/tools/oneapi/components/onemkl.html)
* `CUDABackend` - uses CUDA. This backend runs on GPU. 
* `ArrayFireBackend` - uses [ArrayFire](https://arrayfire.com/)

To use `Holo` Gain, include `autd3/gain/holo.hpp` and the header of each `Backend`.
```cpp
#include "autd3/gain/eigen_backend.hpp"
#include "autd3/gain/holo.hpp"

...

  const auto backend = autd::gain::holo::EigenBackend::create();
  const auto g = autd::gain::holo::SDP::create(backend, foci, amps);
```
The first argument of each algorithm is a `backend`, the second argument is a `vector` of `autd::Vector3` for the position of each focus, and the third argument is a `vector` of `double` for the sound pressure of each focus.
In addition, there are additional parameters for each algorithm.
For details of each parameter, please refer to the respective papers.

If you want to use a `Backend` other than Eigen, you need to compile the respective `Backend` library on your own[^fn_backend].

### BLAS Backend

To build the BLAS backend, set the `BUILD_BLAS_BACKEND` flag on, specify the directory of BLAS library with `BLAS_LIB_DIR`, the directory of BLAS `include` with `BLAS_INCLUDE_DIR`, and the vendor of BLAS with `BLA_VENDO` in CMake,
```
cmake .. -DBUILD_HOLO_GAIN=ON -DBUILD_BLAS_BACKEND=ON -DBLAS_LIB_DIR=<your BLAS library path> -DBLAS_INCLUDE_DIR=<your BLAS include path> -DBLA_VENDOR=<your BLAS vendor>
```
If you use Intel MKL, please set `USE_MKL` flag on,
```
cmake .. -DBUILD_HOLO_GAIN=ON -DBUILD_BLAS_BACKEND=ON -DBLAS_LIB_DIR=<your BLAS library path> -DBLAS_INCLUDE_DIR=<your BLAS include path> -DBLA_VENDOR=Intel10_64lp -DUSE_MKL=ON
```

#### OpenBLAS install example for Windows

Here is an installation example for Windows of [OpenBLAS](https://github.com/xianyi/OpenBLAS), one of the BLAS implementations.
Please also refer to the [Official explanation](https://github.com/xianyi/OpenBLAS/wiki/How-to-use-OpenBLAS-in-Microsoft-Visual-Studio).

First, install Visual Studio 2022 and Anaconda (or miniconda), and open _Anaconda Prompt_.
On the Anaconda Prompt, type the following commands in order.
Note that we install OpenBLAS in `D:/lib/openblas`.
You can set this to any location you like.
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
You may also need to add `%CONDA_HOME%/Library/bin` to your PATH, where `%CONDA_HOME%` is the home directory of Anaconda (or miniconda).

If you follow this installation example, you can build BLAS backend as follows.
```
cmake .. -DBUILD_HOLO_GAIN=ON -DBUILD_BLAS_BACKEND=ON -DBUILD_BLAS_BACKEND=ON -DBLAS_LIB_DIR=D:/lib/openblas -DBLAS_INCLUDE_DIR=D:/lib/openblas/include/openblas -DBLA_VENDOR=OpenBLAS
```

If you get `flangxxx.lib` link errors, you should set Anaconda `lib` directory as `BLAS_DEPEND_LIB_DIR`.
```
cmake .. -DBUILD_HOLO_GAIN=ON -DBUILD_BLAS_BACKEND=ON -DBUILD_BLAS_BACKEND=ON -DBLAS_LIB_DIR=D:/lib/openblas -DBLAS_INCLUDE_DIR=D:/lib/openblas/include/openblas -DBLA_VENDOR=OpenBLAS -DBLAS_DEPEND_LIB_DIR=%CONDA_HOME%/Library/lib
```

### CUDA Backend

To build a CUDA backend, install [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit) and set `BUILD_CUDA_BACKEND` flag in CMake.
```cpp
  cmake .. -DBUILD_HOLO_GAIN=ON -DBUILD_CUDA_BACKEND=ON
```
We have confirmed that it works with CUDA Toolkit version 11.4.100.

### ArrayFire Backend

To build a ArrayFire backend, install [ArrayFire]https://arrayfire.com) and set `BUILD_ARRAYFIRE_BACKEND` flag in CMake.
```cpp
  cmake .. -DBUILD_HOLO_GAIN=ON -DBUILD_ARRAYFIRE_BACKEND=ON
```
We have confirmed that it works with CUDA Toolkit version 3.8.0.

## Grouped

`Grouped` Gain is a `Gain` that allows you to use different `Gain`s for each device when using multiple devices.

In order to use `Grouped` Gain, you should group devices with the third argument of `add_device` in advance.
```cpp
autd->geometry()->add_device(pos1, rot1, 0);
autd->geometry()->add_device(pos2, rot2, 0);
autd->geometry()->add_device(pos3, rot3, 1);
```
In the example above, the first and second devices are in group 0, and the third device is in group 1.

In `Grouped`, you can associate an arbitrary `Gain` to this group number.
```cpp
#include "autd3/gain/eigen_backend.hpp"
#include "autd3/gain/holo.hpp"

...

  const auto g0 = ...;
  const auto g1 = ...;

  const auto g = autd::gain::Grouped::create();
  g->add(0, g0);
  g->add(1, g1);
```
In the above case, group 0 uses `Gain g0` and group 1 uses `Gain g1`.

## Create Custom Gain Tutorial

By inheriting from the `Gain` class, you can create your own `Gain`.
In this section, we will define `Focus` which generates a single focus.

The essence of `Gain` is `vector<array<uint16_t, 249>> _data`, which is a `vector` of the number of devices in the array for the number of transducers of $\SI{16}{bit}$ data.
In each $\SI{16}{bit}$ data, the higher $\SI{8}{bit}$ represents the duty ratio and the lower $\SI{8}{bit}$ represents the phase.
Below is a sample of the `Gain` that generates a single focus.

```cpp
#include "autd3.hpp"
#include "autd3/core/utils.hpp"

class Focus final : public autd::core::Gain {
 public:
  Focus(autd::Vector3 point) : _point(point) {} 
  
  static autd::GainPtr create(autd::Vector3 point) { return std::make_shared<Focus>(point); }
  
  void calc(const autd::GeometryPtr& geometry) override {
    const auto wavenum = 2.0 * M_PI / geometry->wavelength();
    for (size_t dev = 0; dev < geometry->num_devices(); dev++)
      for (size_t i = 0; i < autd::NUM_TRANS_IN_UNIT; i++) {
        const auto dist = (geometry->position(dev, i) - this->_point).norm();
        const auto phase = autd::core::Utilities::to_phase(dist * wavenum);
        this->_data[dev][i] = autd::core::Utilities::pack_to_u16(0xFF, phase);
      }
  }
  
  private:
    autd::Vector3 _point;
};
```

`Controller::send` function takes a `GainPtr` type (an alias of `shared_ptr<autd::core::Gain>`) as an argument.
Therefore, you should define a `create` function to return it.
In the example, since we want to create a single focus, we pass the focus position as an argument.

If you pass `GainPtr` to `Controller::send`, the method `Gain::calc` will be called internally.
So, you should calculate the phase/amplitude in the `calc` method.

For the duty ratio $D \in [0, 255]$ of $\SI{8}{bit}$ specified by SDK and the phase $P \in [0, 255]$ of $\SI{8}{bit}$, the ultrasonic sound pressure $p$ radiated from the transducer will be modeled as,
$$
    p(\br) \propto \sin\left(\frac{\pi}{2}\frac{D + D_\text{offset}}{256} \right)\rme^{\im \frac{2\pi}{\lambda}\|\br\|}\rme^{-2\pi \im \frac{P}{256}}.
$$
Here, $\lambda$ is the wavelength.
Also, $D_\text{offset}$ is $D_\text{offset}=1$ by default, which can be changed optionally.
Therefore, the sound pressure is maximum at $D=255$[^fn_duty].

In order to maximize the sound pressure of the ultrasound emitted from a large number of transducers at a certain point $\bp$, the phase at $\bp$ should be aligned.
Therefore, we have to set $\phi$ as,
$$
    \phi = -2\pi \frac{P}{256} = \frac{2\pi}{\lambda}\|\br\|,
$$
where $r$ is the distance between the transducer and the focal point.

In SDK, you can get the wavelength by `Geometry::wavelength()` and the position of the transducer by `Geometry::position()`.
The first argument of `Geometry::position()` is the index of the device, and the second argument is the local index of transducers.
The `autd::core::Utilities::to_phase` function is a utility function to convert the above phase $\phi$ in $\SI{}{rad}$ to the internal representation $P$ of SDK, defined as follows[^fn_phase].
```cpp
  inline static uint8_t to_phase(const double phase) noexcept {
    return static_cast<uint8_t>(static_cast<int>(std::round((phase / (2.0 * M_PI) + 0.5) * 256.0)) & 0xFF);
  }
```
Also, `autd::core::Utilities::pack_to_u16` is a utility function that simply takes two `uint8_t` values and packs them into the high/low $\SI{8}{bit}$ of the `uint16_t` value.

[^fn_backend]: You need to compile it from the source code. It is not included in the pre-built binary uploaded to GitHub.

[^fn_duty]: In the case of $D_\text{offset}=1$, it seems that the amplitude does not become zero even if $D=0$, but as far as we checked with the oscilloscope, the input signal to the transducer disappears, so there is no problem in fact.

[^fn_phase]: The reason for $+0.5$ is that the return value of `std::arg` etc. is $[-\pi, \pi]$.

[hasegawa2017]: Hasegawa, Keisuke, et al. "Electronically steerable ultrasound-driven long narrow air stream." Applied Physics Letters 111.6 (2017): 064104.

[inoue2015]: Inoue, Seki, Yasutoshi Makino, and Hiroyuki Shinoda. "Active touch perception produced by airborne ultrasonic haptic hologram." 2015 IEEE World Haptics Conference (WHC). IEEE, 2015.

[long2014]: Long, Benjamin, et al. "Rendering volumetric haptic shapes in mid-air using ultrasound." ACM Transactions on Graphics (TOG) 33.6 (2014): 1-10.

[marzo2019]: Marzo, Asier, and Bruce W. Drinkwater. "Holographic acoustic tweezers." Proceedings of the National Academy of Sciences 116.1 (2019): 84-89.

[plasencia2020]: Plasencia, Diego Martinez, et al. "GS-PAT: high-speed multi-point sound-fields for phased arrays of transducers." ACM Transactions on Graphics (TOG) 39.4 (2020): 138-1.

[levenberg1944]: Levenberg, Kenneth. "A method for the solution of certain non-linear problems in least squares." Quarterly of applied mathematics 2.2 (1944): 164-168.

[marquardt1963]: Marquardt, Donald W. "An algorithm for least-squares estimation of nonlinear parameters." Journal of the society for Industrial and Applied Mathematics 11.2 (1963): 431-441.

[madsen2004]: Madsen, Kaj, Hans Bruun Nielsen, and Ole Tingleff. "Methods for non-linear least squares problems." (2004).

[hasegawa2020]: Hasegawa, Keisuke, Hiroyuki Shinoda, and Takaaki Nara. "Volumetric acoustic holography and its application to self-positioning by single channel measurement." Journal of Applied Physics 127.24 (2020): 244904.

[suzuki2021]: Suzuki, Shun, et al. "Radiation Pressure Field Reconstruction for Ultrasound Midair Haptics by Greedy Algorithm with Brute-Force Search." IEEE Transactions on Haptics (2021).

