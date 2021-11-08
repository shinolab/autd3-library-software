# Getting Started

ここでは, 実際にAUTD3を動かす手順について述べる.
なお, 本章ではOSとしてWindows 10 64bitを使用する.
他のOSの場合は適宜読み替えられたい.

## Install 

まず, 必要なツールをインストールする.
本節で使用するツールとversionは以下の通りである.
各々公式の手順に従ってインストールすること.
Visual Studio Community 2019は「C++によるデスクトップ開発」にチェックを入れてインストールすれば良い.
なお, Linuxの場合はgccが, macOSの場合はclangが使えれば良い.
また, 以下はターミナルから操作するため, PATHを通しておくことを推奨する.

* Visual Studio Community 2019 16.11.4
* CMake 3.21.3
* git 2.33.0.windows.2[^fn_git]
* npcap 1.55[^fn_npcap]


## Setup Device

次にデバイスをセットアップする.
ここでは一台のAUTD3のみを使うこととする.
PCのイーサネットポートとAUTD3デバイスのEtherCAT In ([Concept](concept.md)参照) をイーサネットケーブルで接続する.
次に, $\SI{24}{V}$電源を接続する.

### Firmware update

なお, firmwareが古い場合, 動作は保証されない.
本文章におけるfirmwareのversionは1.9が想定される.

firmwareのupdateには[Vivado Design Suite](https://www.xilinx.com/products/design-tools/vivado.html), 及び, [J-Link Software](https://www.segger.com/downloads/jlink/)をインストールしたWindows 10 64bit PCが必要である[^fn_vivado].

まず, AUTD3デバイスとPCを[XILINX Platform Cable](https://www.xilinx.com/products/boards-and-kits/hw-usb-ii-g.html), 及び, [J-Link 9-Pin Cortex-M Adapter](https://www.segger-pocjapan.com/j-link-9-pin-cortex-m-adapter)付きの[J-Link Plus](https://www.segger.com/products/debug-probes/j-link/models/j-link-plus/)で接続し, AUTD3の電源を入れる.
次に, [SDK](https://github.com/shinolab/autd3-library-software)内の`dist/firmware/autd_firmware_writer.ps1`, または, [GitHub Release](https://github.com/shinolab/autd3-library-software/releases)で配布されているパッケージ内の`firmware/autd_firmware_writer.ps1`をpowershellから実行し, 指示に従えばよい.
updateには数分の時間を要する.

## Building first program

まず, ターミナルを開き, 適当なディレクトリを用意する.
```
  mkdir autd3_sample
  cd autd3_sample
```

次に, `CMakeLists.txt`, `main.cpp`ファイルを作成する.
```
└─autd3_sample
        CMakeLists.txt
        main.cpp
```

次に, SDKの最新のバイナリをダウンロードしてくる.
バイナリは[GitHub Release](https://github.com/shinolab/autd3-library-software/releases)にて公開されている.
ダウンロードしたものを解凍して, `include`フォルダと`lib`フォルダを`autd3_sample`フォルダにコピーする.
```
└─autd3_sample
    │  CMakeLists.txt
    │  main.cpp
    ├─include
    └─lib
```

次に, Eigen3をダウンロードしてくる.
Eigen3は行列計算用のヘッダーオンリーライブラリである.
ここでは, カレントディレクトリを`autd3_sample`に変更し, gitのサブモジュールとして追加する.
```
  git init
  git submodule add https://gitlab.com/libeigen/eigen.git eigen
  cd eigen
  git checkout 3.4.0
  cd ..
```
あるいは, 直接[Eigen3](https://gitlab.com/libeigen/eigen)をダウンロードしてきて, `autd3_sample`フォルダ以下に置いても良い. SDKで使用しているversionは3.4.0である.

この時点で, ディレクトリ構成は以下のようになっている.
```
└─autd3_sample
    │  CMakeLists.txt
    │  main.cpp
    ├─include
    ├─lib
    └─eigen
        ├─bench
        ├─blas
        ├─ci
        ├─cmake
        ...
```

次に, `CMakeLists.txt`を以下のようにする.

```
cmake_minimum_required(VERSION 3.16)

project(autd3_sample)
set (CMAKE_CXX_STANDARD 17)

add_executable(main main.cpp)

target_compile_definitions(main PRIVATE _USE_MATH_DEFINES)
target_link_directories(main PRIVATE lib)
target_include_directories(main PRIVATE include eigen)

if(WIN32)
  target_link_directories(main PRIVATE lib/wpcap)
  target_link_libraries(main autd3 soem_link Packet.lib wpcap.lib ws2_32.lib winmm.lib)
elseif(APPLE)
  target_link_libraries(main pcap)
else()
  target_link_libraries(main rt)
endif()

if(WIN32)
    set_property(DIRECTORY \${CMAKE_CURRENT_SOURCE_DIR} PROPERTY VS_STARTUP_PROJECT main)
endif()
```

また, `main.cpp`を以下のようにする.
これは単一焦点に$\SI{150}{Hz}$のAM変調をかける場合のソースコードである.
```cpp
#include <iostream>

#include "autd3.hpp"
#include "autd3/link/soem.hpp"

using namespace std;
using namespace autd;

string get_adapter_name() {
  size_t i = 0;
  const auto adapters = link::SOEM::enumerate_adapters();
  for (auto&& [desc, name] : adapters) cout << "[" << i++ << "]: " << desc << ", " << name << endl;

  cout << "Choose number: ";
  string in;
  getline(cin, in);
  stringstream s(in);
  if (const auto empty = in == "\n"; !(s >> i) || i >= adapters.size() || empty) return "";

  return adapters[i].name;
}

int main() try {
  auto autd = Controller::create();

  autd->geometry()->add_device(Vector3(0, 0, 0), Vector3(0, 0, 0));

  const auto ifname = get_adapter_name();
  auto link = link::SOEM::create(ifname, autd->geometry()->num_devices());
  autd->open(std::move(link));

  autd->clear();

  auto firm_info_list = autd->firmware_info_list();
  for (auto&& firm_info : firm_info_list) cout << firm_info << endl;

  autd->silent_mode() = true;

  const auto focus = Vector3(TRANS_SPACING_MM * ((NUM_TRANS_X - 1) / 2.0), TRANS_SPACING_MM * ((NUM_TRANS_Y - 1) / 2.0), 150.0);
  const auto g = gain::FocalPoint::create(focus);
  const auto m = modulation::Sine::create(150);
  autd->send(g, m);

  cout << "press enter to finish..." << endl;
  cin.ignore();

  autd->close();

  return 0;
} catch (exception& ex) {
  cerr << ex.what() << endl;
}
```

次に, CMakeでbuildする.
```
  mkdir build
  cd build
  cmake .. -G "Visual Studio 16 2019" -A x64
```
これで, buildディレクトリ以下に`autd3_sample.sln`が生成されているはずなので, これを開き, mainプロジェクトを実行する.
なお, 実行に際して, Visual StudioのConfigurationをDebugからReleaseに変更すること.
また, Linux/macOSの場合は, 実行時にroot権限が必要な場合がある.

## Explanation

特に解説は不要と思われるが, 念の為上記コードの解説をしておく.

SDKを使用するには, `autd3.hpp`ヘッダーをインクルードする.
また, `link::SOEM`を使用するために`autd3/link/soem.hpp`ヘッダーも必要となる.
```cpp
#include "autd3.hpp"
#include "autd3/link/soem.hpp"
```

次に, `Controller`の作成をする.
```cpp
  auto autd = autd::Controller::create();
```

その後, デバイスの配置を指定する.
```cpp
  autd->geometry()->add_device(autd::Vector3(0, 0, 0), autd::Vector3(0, 0, 0));
```
`add_device`の第一引数は位置, 第2引数は回転を表す.
位置は自分の設定したグローバル座標系におけるデバイスの原点を指定する.
また, 回転はZYZのオイラー角, または, クオータニオンで指定する.
ここでは, 回転も移動も行わないものとする.

次に, `Link`を作成し, デバイスと接続する.
```cpp
  const auto ifname = get_adapter_name();
  auto link = link::SOEM::create(ifname, autd->geometry()->num_devices());
  autd->open(std::move(link));
```
`link::SOEM::create()`の第一引数はインターフェース名で, 第2引数は接続しているAUTD3デバイスの数である. 
インターフェース名は, Windowsの場合は, `\Device\NPF_{XXXXXXXX-XXXX-XXXX-XXXX-XXXXXXXXXXXX}`のような文字列となり, 確認するのはひと手間かかるので, ユーティリティ関数`get_adapter_name`を用意した. 実行時に適当なものを選択されたい.
(macやLinuxの場合は`ifconfig`等で確認することもできる.)
また, linkの型は`unique_ptr`であるため, `Controller`に渡す際は`move`する必要がある.

次に, AUTDデバイスの初期化を行う.
電源投入時に初期化されるので`clear()`は呼ばなくても良いかもしれない.
```cpp
  autd->clear();
```

次に, firmwareのバージョンを確認している.
これは必須ではない.
```cpp
  auto firm_info_list = autd->firmware_info_list();
  for (auto&& firm_info : firm_info_list) cout << firm_info << endl;
```

次に, silentモードを指定する.
```cpp
  autd->silent_mode() = true;
```
デフォルトではONになっているので, これも実際には呼ぶ必要はない.
OFFにしたい場合は`false`を与える.
silentモードでは, 振動子に与える位相/振幅パラメータをLow-pass filterに通すことで, 静音化を行う.

その後, 単一焦点を表す`Gain`と$\SI{150}{Hz}$のSin波変調をかける`Modulation`を作成デバイスに送信する.
```cpp
  const auto point = autd::Vector3(autd::TRANS_SPACING_MM * ((autd::NUM_TRANS_X - 1) / 2.0), autd::TRANS_SPACING_MM * ((autd::NUM_TRANS_Y - 1) / 2.0), 150.0);
  const auto g = autd::gain::FocalPoint::create(point);
  const auto m = autd::modulation::Sine::create(150);
  autd->send(g, m);
```
`point`が若干ややこしいが, `TRANS_SPACING_MM`が振動子の配置間隔を, `NUM_TRANS_X`, `NUM_TRANS_Y`がそれぞれ, $x,y$軸方向の振動子の数を表している.
そのため, `pointは`振動子アレイの中心から直上$\SI{150}{mm}$を表す.

最後に, デバイスとの接続を切ってお終いである.
```cpp
  autd->close();
```

次頁では基本的な関数について解説していく.
なお, [オンラインのAPIドキュメント](https://shinolab.github.io/autd3-library-software/index.html)もあるのでそちらも参考にする事.

[^fn_git]: 動かすのに必須ではないが, 作業の単純化のため使用

[^fn_npcap]: SOEM linkを使用するのに使う. TwinCAT linkの場合は必要ない.

[^fn_vivado]: Vivadoは最小構成でも数十GBの容量を食うので, 適当なPCを用意しておくことをおすすめする.
