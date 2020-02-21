# README #

## Windows ##

run `build.ps1`

### Option ###

[] is a default.

* -BUILD_DIR = [\build]
* -NOUNITY = [False]
* -VS_VERSION = 2017, [2019]
* -ARCH = [x64]
* -DISABLE_MATLAB = [False]
* -ENABLE_TEST = [False]
* -TOOL_CHAIN = [""]

### 注意: VS2017 ###

AUTDSharpのプロパティの「ビルド」→「詳細設定」で言語バージョンをC#7.2以上にして下さい.

cmakeのバグ(?)でマイナーバージョンの指定ができないためです.

### 注意: 単体テスト ###

vcpkgを利用します.
vcpkgでgtestをインストールしておく必要があります.

また, vcpkgのtoolchain fileを指定が要ります.

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

## Memo ##
* cmakeのバグかは知らないけど, -Aオプションでx64プラットフォームを指定してもC#のターゲットがx64にならない.
そのため, csproj内を直接書き換えることにしている

## Coding Style ##

Google C++ Style Guide with LineLength = 150

# Author #

Shun Suzuki, 2019