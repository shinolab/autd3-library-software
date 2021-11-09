# Emulator

[autd-emulator](https://github.com/shinolab/autd-emulator)はその名の通り, クロスプラットフォームで動作するAUTD3のエミュレータである.

## Install

Windows 10 64bit版のみコンパイル済みのバイナリが[GitHubで配布されている](https://github.com/shinolab/autd-emulator/releases)のでこれをダウンロードして実行すれば良い.

その他の場合はRustのコンパイラをインストールして, 各自コンパイルすること.
```
git clone https://github.com/shinolab/autd-emulator.git
cd autd-emulator
cargo run --release
```

## How to

<figure>
  <img src="../../fig/Users_Manual/emu-home.jpg"/>
  <figcaption>Emulator</figcaption>
</figure>

autd-emulatorを実行すると上図のような画面になる.
この状態で, Emulator linkを使用したクライアントプログラムを実行すると, クライアントプログラムの内容に合わせた音場が表示される.
図の中央の黒いパネルをSliceと呼び, このSliceを使って任意の位置の音場を可視化できる.
また, 振動子の位相が色相で, 振幅が色強度で表される.

なお, エミュレータで表示される音場はシンプルな球面波の重ね合わせであり, 指向性や非線形効果などは考慮されない.

画面左に表示されるGUIでSliceやカメラの操作が行える.
なお, GUIには[Dear ImGui](https://github.com/ocornut/imgui)を用いており, マウスによる操作のほか, "Ctrl+クリック"で数値入力モードになる.

また, GUI以外の場面の"ドラッグ"でカメラの移動, "Shift+ドラッグ"でカメラの回転が行える.

### Slice tab

SliceタブではSliceの大きさと位置, 回転を変えられる.
回転はXYZのオイラー角で指定する.
なお, "xy", "yz", "zx"ボタンを押すと, Sliceを各平面に平行な状態に回転させる.

Sliceでは音圧の強さを色で表現する.
Color scaleはこの色空間の音圧の最大値を表す.
大量のデバイスを使用すると色が飽和する場合があるので, その時はColor scaleの値を大きくすれば良い.
また, Sliceそのもののアルファ値をSlice alphaで指定できる.

さらに, `offscreen_renderer feature`をenableにした状態でコンパイルすると, Sliceに表示されている音場の保存及び録画ができるようになる[^1].

### Camera tab

Sliceタブではカメラの位置, 回転, Field of Viewの角度, Near clip, Far clipの設定を変えられる.
回転はXYZのオイラー角で指定する.

### Config tab

Configタブでは波長と振動子のアルファ値, 及び, 背景色の設定ができる.

また, Emulator linkと接続した後は, 各デバイスごとの表示/イネーブルを切り替えられる.
表示をOffにした場合は, 表示されないだけで音場に寄与する.
イネーブルをOffにすると音場に寄与しなくなる.
さらに, デバイス毎の軸の表示もできる.

### Info tab

InfoタブではModulationやSequenceの情報が確認できる.

ModulationはSlice上には反映されない.
代わりに, 音圧がどのように変調されるかがこのタブで表示される.
また, rawモードではDuty比がどのように変調されるかが表示される.

Sequenceを送信した場合は, Sequenceの情報が表示される.
Sequenceは自動的に切り替わったりしない, 代わりにSequence idxで何番目のデータを表示するかを指定する.

flagの項ではControl flagが表示される.
Silent modeとForce fan以外は内部で使われるものなので無視する.
なお, Silent modeをOnにしてもSlice上の表示には変化はない.

また, Output delayとDuty offsetも無視される.

### Log tab

Logタブではデバッグ用のログが表示される.

### Other settings

すべての設定は`settings.json`に保存される.
幾つかの設定は`settings.json`からのみ編集できる.
この中で重要なものとして, portとvsyncがある.

portはSDKのEmulator linkとの間で使うポート番号である.
また, vsyncをtrueにすると垂直同期が有効になる.

[^1]: Windows 10 64bit向けのコンパイル済みのバイナリではONになっている. なお, この機能は Vulkanのラッパーである[Vulkano](https://github.com/vulkano-rs/vulkano)を使用している, コンパイルの際はVulkanoのコンパイル方法も確認すること.
