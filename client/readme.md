# README #

build.ps1またはbuild.batを使うとbuild以下にautd.sln生成されます.

## Option (PowerShell) ##

[]内がデフォルト指定.

* -BUILD_DIR = [\build]
* -NOUNITY = [False]
* -VS_VERSION = 2017, [2019]
* -ARCH = [x64]
* -ENABLE_TEST = [False]
* -TOOL_CHAIN = [""]

## Option (CMD) ##

デフォルトではVS2019にってます.

* -vs2017 : Visual Studio 2017にする.
* -x86: 32bitを使う人用.
* -nounity: unityいらない人用.
* -test: 単体テストプロジェクトを含む場合.

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




## Memo ##
cmakeのバグかは知らないけど, -Aオプションでx64プラットフォームを指定してもC#のターゲットがx64にならない.

そのため, csproj内を直接書き換えることにしている.

# Author #

Shun Suzuki, 2019