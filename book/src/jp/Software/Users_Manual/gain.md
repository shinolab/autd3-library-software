# Gain

AUTDは各振動子の位相/振幅を個別に制御することができ, これによって様々な音場を生成できる.
`Gain`はこれを管理するクラスであり, SDKにはデフォルトでいくつかの種類の音場を生成するための`Gain`がデフォルトでいくつか用意されている.

## FocalPoint

`FocalPoint`は最も単純な`Gain`であり, 単一焦点を生成する.
```cpp
    const auto g = autd::gain::FocalPoint::create(autd::Vector3(x, y, z));
```
`FocalPoint::create`の第1引数には焦点の位置を指定する.
第2引数として, 振幅をDuty比 (`uint8_t`), または, 0-1の規格化された音圧振幅 (`double`) で指定できる.

ここでDuty比$D$と音圧$p$の関係性について注意する.
理論的には
$$
    p \propto \sin \pi D
$$
の関係性がある.
そのため, 音圧はDuty比$0$で最小値$p=0$, Duty比$50\,\%$で最大値を取るが, その間の関係式は線形ではない.
音圧振幅で指定する場合, $p=1$を最大値として, 内部で上記の式の逆変換を行いDuty比に変換される.

## BesselBeam

`BesselBeam`ではその名の通りBessel beamを生成する.
この`Gain`は長谷川らの論文[hasegawa2017]に基づく.
```cpp
  const autd::Vector3 apex(x, y, z);
  const autd::Vector3 dir = autd::Vector3::UnitZ();
  const double theta_z = 0.3;
  const auto g = autd::gain::BesselBeam::create(apex, dir, theta_z);
```

第1引数はビームを生成する仮想円錐の頂点であり, 第2引数はビームの方向, 第3引数はビームに垂直な面とビームを生成する仮想円錐の側面となす角度である (下図の$\theta_z$).
第4引数として, 振幅をDuty比 (`uint8_t`), または, 0-1の規格化された音圧振幅 (`double`) で指定できる.

<figure>
  <img src="https://raw.githubusercontent.com/shinolab/autd3-library-software/master/book/src/fig/Users_Manual/1.4985159.figures.online.f1.jpg"/>
  <figcaption>Bessel beam ([hasegawa2017]より引用)</figcaption>
</figure>

## PlaneWave

`PlaneWave`は平面波を生成する.
```cpp
    const auto g = autd::gain::PlaneWave::create(autd::Vector3(x, y, z));
```
`PlaneWave::create`の第1引数には平面波の方向を指定する.
第2引数として, 振幅をDuty比 (`uint8_t`), または, 0-1の規格化された音圧振幅 (`double`) で指定できる.

## TransducerTest

`TransducerTest`はデバッグ用の`Gain`であり, ある一つの振動子のみを駆動する.
```cpp
    const auto g = autd::gain::TransducerTest::create(index, duty, phase);
```
`TransducerTest::create`の第1引数には振動子のindex, 第2引数はDuty比, 第3引数には位相を指定する.

## Null

`Null`は振幅0の`Gain`である.
```cpp
    const auto g = autd::gain::Null::create();
```

## Holo (Multiple foci)

`Holo`は多焦点を生成するための`Gain`である.
多焦点を生成するアルゴリズムが幾つか提案されており, SDKには以下のアルゴリズムが実装されていてる.

* `SDP` - Semidefinite programming, 井上らの論文[inoue2015]に基づく
* `EVD` - Eigen value decomposition, Longらの論文[long2014]に基づく
* `Naive` - 単一焦点解の重ね合わせ
* `GS` - Gershberg-Saxon, Marzoらの論文[marzo2019]に基づく
* `GSPAT` - Gershberg-Saxon for Phased Arrays of Transducers, Plasenciaらの論文[plasencia2020]に基づく
* `LM` - Levenberg-Marquardt, LM法は[levenberg1944,marquardt1963]で提案された非線形最小二乗問題の最適化法, 実装は[madsen2004]に基づく.
* `GaussNewton` - Gauss-Newton法
* `GradientDescent` - Gradient descent法
* `APO` - Acoustic Power Optimization, 長谷川らの論文[hasegawa2020]に基づく
* `Greedy` - Greedy algorithm and Brute-force search, 鈴木らの論文[suzuki2021]に基づく

また, 各手法は計算Backendを選べるようになっている.
SDKには以下の`Backend`が用意されている

* `EigenBackend` - [Eigen](https://eigen.tuxfamily.org/index.php?title=Main_Page)を使用, デフォルトで利用可能
* `BLASBackend` - [OpenBLAS](https://www.openblas.net/)や[Intel MKL](https://software.intel.com/content/www/us/en/develop/tools/oneapi/components/onemkl.html)等のBLAS/LAPACKを使用
* `CUDABackend` - CUDAを使用, GPUで実行
* `ArrayFireBackend` - [ArrayFire](https://arrayfire.com/)を使用

`Holo`を使用するには`autd3/gain/holo.hpp`と各`Backend`のヘッダーを`include`する.
```cpp
#include "autd3/gain/eigen_backend.hpp"
#include "autd3/gain/holo.hpp"

...

  const auto backend = autd::gain::holo::EigenBackend::create();
  const auto g = autd::gain::holo::SDP::create(backend, foci, amps);
```
各アルゴリズムの第1引数は`backend`, 第2引数は各焦点の位置を`autd::Vector3`の`vector`で, 第3引数は各焦点の音圧を`double`の`vector`で指定する.
また, 各アルゴリズムごとに追加のパラメータが存在する.
各パラメータの詳細はそれぞれの論文を参照されたい.

また, Eigen以外の`Backend`を使用するには, それぞれの`Backend`ライブラリをコンパイルする必要がある[^fn_backend].

### BLAS Backend

BLASバックエンドをビルドするには, CMakeの`BUILD_BLAS_BACKEND`フラグをOnにして, `BLAS_LIB_DIR`でBLASライブラリのディレクトリを, `BLAS_INCLUDE_DIR`でBLASの`include`ディレクトリを指定し, `BLA_VENDO`でBLASのベンダを指定する,
```cpp
cmake .. -DBUILD_HOLO_GAIN=ON -DBUILD_BLAS_BACKEND=ON -DBLAS_LIB_DIR=<your BLAS library path> -DBLAS_INCLUDE_DIR=<your BLAS include path> -DBLA_VENDOR=<your BLAS vendor>
```
なお, Intel MKLを使用する場合は更に, `USE_MKL`フラグをOnにする.
```cpp
cmake .. -DBUILD_HOLO_GAIN=ON -DBUILD_BLAS_BACKEND=ON -DBLAS_LIB_DIR=<your BLAS library path> -DBLAS_INCLUDE_DIR=<your BLAS include path> -DBLA_VENDOR=Intel10_64lp -DUSE_MKL=ON
```

#### OpenBLAS install example for Windows

ここではWindows向けに, BLAS実装の一つである[OpenBLAS](https://github.com/xianyi/OpenBLAS)のインストール例を載せておく.
[Officialの説明](https://github.com/xianyi/OpenBLAS/wiki/How-to-use-OpenBLAS-in-Microsoft-Visual-Studio)もあるのでこちらも参考にすること.
なお, OpenBLASのversion 0.3.18での動作を確認している.

まず, Visual Studio 2022とAnaconda (または, miniconda)をインストールし, Anaconda Promptを開く.
Anaconda Prompt上で以下のコマンドを順に打っていく.
なお, ここでは`D:/lib/openblas`にOpenBLASをインストールすることにしている.
これは, 各自好きな場所に設定されたい.
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
また, `%CONDA_HOME%/Library/bin`をPATHに追加する必要があるかもしれない.
ここで, `%CONDA_HOME%`はAnaconda (または, miniconda) のホームディレクトリである.

このインストール例に従った場合は
```cpp
cmake .. -DBUILD_HOLO_GAIN=ON -DBUILD_BLAS_BACKEND=ON -DBUILD_BLAS_BACKEND=ON -DBLAS_LIB_DIR=D:/lib/openblas -DBLAS_INCLUDE_DIR=D:/lib/openblas/include/openblas -DBLA_VENDOR=OpenBLAS
```
とすれば良い.

`flangxxx.lib`関連のリンクエラーが発生する場合は, さらに`BLAS_DEPEND_LIB_DIR`オプションでAnacondaの`lib`フォルダを指定する.
```cpp
cmake .. -DBUILD_HOLO_GAIN=ON -DBUILD_BLAS_BACKEND=ON -DBUILD_BLAS_BACKEND=ON -DBLAS_LIB_DIR=D:/lib/openblas -DBLAS_INCLUDE_DIR=D:/lib/openblas/include/openblas -DBLA_VENDOR=OpenBLAS -DBLAS_DEPEND_LIB_DIR=%CONDA_HOME%/Library/lib
```

### CUDA Backend

CUDAバックエンドをビルドする場合, [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit)をインストールし, CMakeで`BUILD_CUDA_BACKEND`フラグをOnにすれば良い.
```
  cmake .. -DBUILD_HOLO_GAIN=ON -DBUILD_CUDA_BACKEND=ON
```
なお,  CUDA Toolkit version 11.5.50で動作を確認している.

### ArrayFire Backend

ArrayFireバックエンドをビルドする場合, [ArrayFire](https://arrayfire.com/)をインストールし[^fn_af], CMakeで`BUILD_ARRAYFIRE_BACKEND`フラグをOnにすれば良い.
```
  cmake .. -DBUILD_HOLO_GAIN=ON -DBUILD_ARRAYFIRE_BACKEND=ON
```
なお, ArrayFire version 3.8.0で動作を確認している.

## Grouped

`Grouped`は複数のデバイスを使用する際に,
各デバイスで別々の`Gain`を使用するための`Gain`である.

`Grouped`では, デバイスIDと任意の`Gain`を紐付けて使用する.
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
上の場合は, デバイス0が`Gain g0`, デバイス1が`Gain g1`を使用する.

## Create Custom Gain Tutorial

`Gain`クラスを継承することで独自の`Gain`を作成することができる.
ここでは, `FocalPoint`と同じように単一焦点を生成する`Focus`を実際に定義してみることにする.

`Gain`の実態は`vector<array<Drive, 249>> _data`であり, 位相/振幅データを格納する`Drive`構造体の振動子数分の配列のデバイス数分の`vector`になっている.
下が, 単一焦点を生成する`Gain`のサンプルである.

```cpp
#include "autd3.hpp"
#include "autd3/core/utils.hpp"

class Focus final : public autd::core::Gain {
 public:
  explicit Focus(const autd::Vector3 point) : _point(point) {}

  static autd::GainPtr create(autd::Vector3 point) { return std::make_shared<Focus>(point); }

  void calc(const autd::Geometry& geometry) override {
    const auto wavenum = 2.0 * M_PI / geometry.wavelength();
    for (const auto& device : geometry)
      for (const auto& transducer : device) {
        const auto dist = (transducer.position() - this->_point).norm();
        this->_data[device.id()][transducer.id()].duty = 0xFF;
        this->_data[device.id()][transducer.id()].phase = autd::core::utils::to_phase(dist * wavenum);
      }
  }

 private:
  autd::Vector3 _point;
};
```

`Controller::send`関数は`GainPtr`型 (`shared_ptr<autd::core::Gain>`のエイリアス) を引数に取る.
そのため, これを返すような`create`関数を定義しておく.
今回は, 単一焦点を生成するので, 焦点位置を引数で渡してある.

`Controller::send`に`GainPtr`を渡すと, 内部で`Gain::calc`メソッドが呼ばれる.
そのため, この`calc`メソッド内で位相/振幅の計算を行えば良い.

SDKで指定した$\SI{8}{bit}$のDuty比$D \in [0, 255]$, 及び, $\SI{8}{bit}$の位相$P \in [0, 255]$に対して, 振動子から放射される超音波音圧は$p$
$$
    p(\br) \propto \sin\left(\frac{\pi}{2}\frac{D + D_\text{offset}}{256} \right)\rme^{\im \frac{2\pi}{\lambda}\|\br\|}\rme^{-2\pi \im \frac{P}{256}}
$$
のようにモデル化されている.
ここで, $\lambda$は波長である.
また, $D_\text{offset}$はデフォルトで$D_\text{offset}=1$であり, これはオプションで変更可能である.
したがって, $D=255$で音圧は最大となる[^fn_duty].

ある点$\bp$で多数の振動子からの放出された超音波の音圧が最大になるためには, $\bp$での位相が揃えば良い.
したがって, 上の式を見れば明らかなように,
$$
    \phi = -2\pi \frac{P}{256} = \frac{2\pi}{\lambda}\|\br\|
$$
とすれば良い.
ここで, $r$は振動子と焦点位置との間の距離である.

SDKでは, 波長は`Geometry::wavelength()`, できる.
Geometryにはイテレータが定義されており, `Device`のイテレータが返される.
また, `Device`にもイテレータが定義されており, `Transducer`のイテレータが返され, ここから振動子の位置を取得できる.
また, `autd::core::Utilities::to_phase`関数は, 上記の$\SI{}{rad}$単位の位相$\phi$をSDKの内部表現$P$に変換するためのユーティリティ関数で, 以下のように定義されている[^fn_phase].
```cpp
  inline static uint8_t to_phase(const double phase) noexcept {
    return static_cast<uint8_t>(static_cast<int>(std::round((phase / (2.0 * M_PI) + 0.5) * 256.0)) & 0xFF);
  }
```

[^fn_backend]: 各自ソースコードからコンパイルする必要がある. GitHubにアップロードされているpre-built binaryには含まれていない.

[^fn_af]: `%AF_PATH%/lib`をPATHに追加する必要があるかもしれない, ここで`%AF_PATH%`はArrayFire のインストールディレクトリである.

[^fn_duty]: $D_\text{offset}=1$の場合, $D=0$でも振幅が0にならないように思われるが, オシロで確認した限り振動子への入力信号は消えるので事実上問題ない.

[^fn_phase]: $+0.5$しているのは, `std::arg`などの返り値が$[-\pi, \pi]$だからである. 符号ありの`int`にしているので実際には不要だと思うが.

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

