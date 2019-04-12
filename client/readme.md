# README #

build.batを使うとbuild以下にautd.sln生成されます.

## Option ##

デフォルトではVisual Studio 2019を使うようになっています.

* -vs2017 : Visual Studio 2017にする. (以下の注意を参照)
* -x86: 32bitを使う人用.
* -nounity: unityいらない人用.
* -test: 単体テストプロジェクトを含む場合. (以下の注意を参照)

### 注意: VS2017 ###

AUTDSharpのプロパティの「ビルド」→「詳細設定」で言語バージョンをC#7.2以上にして下さい.

cmakeのバグでマイナーバージョンの指定ができないためです.

### 注意: 単体テスト ###

vcpkgを使います. vcpkgでgtestをinstallしておいて下さい.

また, -testに続いて, vcpkgのtoolchain fileを指定して下さい.

EX.
```
 -test "-DCMAKE_TOOLCHAIN_FILE=C:[...]\vcpkg\scripts\buildsystems\vcpkg.cmake"
```
## Memo ##
cmakeのバグかは知らないけど, -Aオプションでx64プラットフォームを指定してもC#のターゲットがx64にならない.

そのため, csproj内を直接書き換えることにしている.

# Author #

Shun Suzuki, 2019