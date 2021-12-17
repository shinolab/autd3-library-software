# Controller

ここでは, Controllerクラスに存在するその他の機能を紹介する.

## Output enable

出力イネーブルの設定を行う.
```cpp
  autd.output_enable() = false;
```
FPGAの出力はこのフラグとの論理積になる.

実際にフラグが更新されるのは[Send functions](#send-functions)のどれかを呼び出し後になる.

## Silent mode

AMやSpatio-Temporal Modulationにおいて, 位相/振幅の急激な変化が起こると, ノイズが発生する.
SDKにはこれを抑制するためのフラグが用意されている.
```cpp
  autd.silent_mode() = true;
```
このフラグをOnにすると, デバイスの内部で位相/振幅データにLow-pass filterが適用され, 位相/振幅の変化がなめらかになりノイズが抑制される[suzuki2020].

実際にフラグが更新されるのは[Send functions](#send-functions)のどれかを呼び出し後になる.

## Output balance

Hardware設計の都合により, AUTD3は振動子の出力を印加していない状態, 即ち, `pause`または`stop`を呼んだ後の状態において, 振動子には$\SI{-12}{V}$の電圧が印加されている.
`output_balance`フラグをOnにすると, これを基準電位に落とすことができる.

実際にフラグが更新されるのは[Send functions](#send-functions)のどれかを呼び出し後になる.

```cpp
  autd.output_balance() = true;
```

ただし, この機能はFPGAからの出力を高速にOn/Off切り替えることで実現している.
ハーフブリッジドライバへの影響は調べていないので, 使用には注意すること.
基本的に, 使わないときはそもそも電源を落としておくことをおすすめする.

<figure>
  <img src="https://raw.githubusercontent.com/shinolab/autd3-library-software/master/book/src/fig/Users_Manual/no_balance.jpg"/>
  <figcaption>Without balancing</figcaption>
</figure>

<figure>
  <img src="https://raw.githubusercontent.com/shinolab/autd3-library-software/master/book/src/fig/Users_Manual/balance.jpg"/>
  <figcaption>With balancing</figcaption>
</figure>

## Check Ack

`check_ack`フラグをOnにすると, デバイスへのデータ送信時に, 送信データがきちんとデバイスで処理されたかどうかを確認するようになる.
```cpp
  autd.check_ack() = true;
```
`check_ack`が`true`の場合, デバイスにデータを送信する関数 ([Send functions](#send-functions)) は, 送信データがきちんとデバイスで処理されたかどうかを返すようになる.

基本的にOffで問題ないはずだが, 確実にデータを送りたいときにはOnにする.
なお, Onにすると[Send functions](#send-functions)の実行時間は増加する.

## Force fan

AUTD3デバイスには温度計が搭載されており, 温度が高くなると自動的にファンが起動する.
`force_fan`フラグはこのファンを強制的に起動するためのフラグである.
実際にフラグが更新されるのは[Send functions](#send-functions)のどれかを呼び出し後になる.

```cpp
  autd.force_fan() = true;
```

なお, 強制的にONにすることはできるが, 強制的にOFFにすることはできない.
ファンを使いたくない場合は, 物理的に配線を抜いておくこと.

## Read FPGA info

`reads_fpga_info`フラグをONにすると, デバイスがFPGAの状態を返すようになる.
実際にフラグが更新されるのは[Send functions](#send-functions)のどれかを呼び出し後になる.

FPGAの状態は`fpga_info`関数で取得できる.
```cpp
  autd.reads_fpga_info() = true;
  autd.update_ctrl_flag();
  const auto fpga_info = autd.fpga_info();
```
`fpga_info`の返り値は`FPGAInfo`のデバイス分だけの`vector`である.
`FPGAInfo::is_running_fan`でファンが起動しているかどうかを確認できる.

## Duty offset

$D_\text{offset}$ ([Create Custom Gain Tutorial](gain.md#create-custom-gain-tutorial)参照) を変更するには`autd::DelayOffsets`構造体を送信する.
なお, `offset`は下位$\SI{1}{bit}$のみが用いられる.
したがって, $D_\text{offset}=0,1$のみ使用できる.

```cpp
  autd::DelayOffsets delay_offsets(autd.geometry().num_devices());

  delay_offsets[0].offset = 0;  // duty offset is 0 for 0-th transducer 
  autd << delay_offsets;       // apply change
```
 
## Output delay 

各振動子の出力を$\SI{25}{μs}$単位で相対的に遅らせることができる.
これには, `autd::DelayOffsets`構造体を送信する.
ただし, Delay値は下位$\SI{7}{bit}$のみ使用され, 遅延は最大で$127=\SI{3.175}{ms}$である.

```cpp
  autd::DelayOffsets delay_offsets(autd.geometry().num_devices());

  delay_offsets[0].delay = 4;  // 4 cycle = 100 us delay in 0-th transducer
  autd << delay_offsets;       // apply change
```

## pause/resume/stop

`pause`関数を呼び出すと出力を一時停止する.
また, `resume`関数で再開する.

`stop`関数も出力を止めるが, `resume`で再開できない.

`pause`関数は出力を瞬間的に停止する (具体的にはFPGAからの出力に0との論理積をとる) ので, ノイズが発生することがある.
`stop`関数はそれを抑えるようになっている.

## clear

デバイス内のフラグや`Gain`/`Modulation`データ等をクリアする.

## Firmware information

`firmware_info_list`関数でFirmwareのバージョン情報を取得できる.

```cpp
 for (auto&& firm_info : autd.firmware_info_list()) std::cout << firm_info << std::endl;
```

## Send functions

Send functionsとは, 実際にデバイスにデータを送信する関数の総称である.
これらの関数を呼び出すことで, `output enable`, `silent mode`, `force fan`, `reads FPGA info`, `output balance`のフラグが更新される.
また, これらの関数は`check_ack`フラグによって挙動が変わる.
`check_ack`が`true`の場合, これらの関数はデバイスが実際にデータを処理するまで待機する.
特に, `Modulation`/`Sequence`を送信する際は1フレーム毎に確認が入るので, 処理時間が大きく増加する可能性がある.
また, デバイスがデータを処理したことを確認できなかった場合に`false`を返してくる.
`check_ack`が`false`の場合はデータが処理されたかどうかを確認しない, また, 返り値は常に`true`になる.

送信系の関数の一覧は次のとおりである.

* `update_ctrl_flag`
* `clear`[^fn_clear]
* `close`
* `stop`
* `pause`
* `resume`
* `send`
* `<<` (stream operator)

[^fn_clear]: フラグもクリアされる

[suzuki2020]: Suzuki, Shun, et al. "Reducing amplitude fluctuation by gradual phase shift in midair ultrasound haptics." IEEE transactions on haptics 13.1 (2020): 87-93.
