# Sequence

SDKでは, `Gain`を周期的に切り替えるための機能が用意されている.
これは, `stm`と`Sequence`という機能に大別される.
前者は任意の`Gain`をホスト側のメモリの許す限り使用することができるが, Softwareのタイマで実行されるため, 時間的な精度が低い.
後者はHardwareのタイマで実行されるため, 精度が高いが制約が強い.
以下ではまず前者の`stm`について述べ, 後者の`Sequence`はその次に述べる.

## stm

以下は, `stm`を用いて, 単一焦点を周期的に円周上で動かすサンプルである.
```cpp
  auto autd = autd::Controller::create();
  
  ...
  
  const auto stm = autd.stm();

  const autd::Vector3 center(x, y, z);
  constexpr auto point_num = 100;
  for (auto i = 0; i < point_num; i++) {
    constexpr auto radius = 20.0;
    const auto theta = 2.0 * M_PI * static_cast<double>(i) / point_num;
    const autd::Vector3 pos(radius * cos(theta), radius * sin(theta), 0.0);
    autd::gain::FocalPoint g(center + pos);
    stm << g;
  }

  stm.start(0.5);  // 0.5 Hz

  std::cout << "press any key to stop..." << std::endl;
  std::cin.ignore();

  stm.stop();
  stm.finish();
```
上記の例だと, `center`を中心に半径$\SI{20}{mm}$の円周上を等間隔に100点分サンプリングし, その点を一周$\SI{0.5}{Hz}$の周波数で回す.

`stm`を使用するには, `Controller::stm`で`stm`用のコントローラを取得する.
この`stm`コントローラに`Gain`を追加していく.
最後に, `start`関数で`stm`を開始する.
`stm`を一時停止する場合は`stop`関数を呼ぶ.
`stop`の後にもう一度`start`を呼べば再開される.
`stm`を終了する場合は`finish`関数を呼び出す.

`stm`コントローラを取得してから`finish`を呼び出すまでの間は, 元の`Controller`の使用は禁止される.

## Sequence

`Sequence`はHardwareのタイマでSpatio-Temporal Modulationを実現する.
SDKには単一焦点のみをサポートする`PointSequence`と任意の`Gain`をサポートする`GainSequence`が用意されている.

### PointSequence

`PointSequence`には以下の制約がある.
* 最大サンプリング点は65536
* サンプリング周波数は$\SI{40}{kHz}/N, N=1,2,...,65536$

`PointSequence`の使用方法は`stm`のサンプルとほぼ同じである.
```cpp
  autd::PointSequence seq;

  const autd::Vector3 center(x, y, z);
  constexpr auto point_num = 200;
  for (auto i = 0; i < point_num; i++) {
    constexpr auto radius = 30.0;
    const auto theta = 2.0 * M_PI * static_cast<double>(i) / static_cast<double>(point_num);
    const autd::Vector3 p(radius * std::cos(theta), radius * std::sin(theta), 0);
    seq << center + p;
  }

  const auto actual_freq = seq.set_frequency(1);
  std::cout << "Actual frequency is " << actual_freq << " Hz\n";
  autd << seq;
```

サンプリング点数とサンプリング周期に関する制約によって, 指定した周波数と実際の周波数は異なる可能性がある.
例えば, 上記の例は, 200点を$\SI{1}{Hz}$で回すため, サンプリング周波数は$\SI{200}{Hz}=\SI{40}{kHz}/200$とすればよく, 制約を満たす.
しかし, `point_num`=199にすると, サンプリング周波数を$\SI{199}{Hz}$にしなければならないが, $\SI{199}{Hz}=\SI{40}{kHz}/N$を満たすような$N=1,2,...,65535$は存在しない, そのため, 最も近い$N$が選択される.
これによって, 指定した周波数と実際の周波数がずれる.
`set_frequency`関数はこの実際の周波数を返してくる.

### GainSequence

`GainSequence`は任意の`Gain`を扱えるが, 代わりに使用できる`Gain`の個数が2048個に減る.

`GainSequence`の使用サンプルは`PointSequence`とほぼ同じである.
```cpp
  autd::GainSequence seq(autd.geometry());

  const autd::Vector3 center(x, y, z);
  constexpr auto point_num = 200;
  for (auto i = 0; i < point_num; i++) {
    constexpr auto radius = 30.0;
    const auto theta = 2.0 * M_PI * static_cast<double>(i) / static_cast<double>(point_num);
    const autd::Vector3 p(radius * std::cos(theta), radius * std::sin(theta), 0);
    autd::gain::FocalPoint g(center + p);
    seq << g;
  }

  const auto actual_freq = seq.set_frequency(1);
  std::cout << "Actual frequency is " << actual_freq << " Hz\n";
  autd << seq;
```
周波数の制約も`PointSequence`と同じである.

`GainSequence`は位相/振幅データをすべて送信するため, レイテンシが大きい[^fn_gain_seq].
これを削減するために, 位相のみを送る倍速モードと, 位相のみを$\SI{4}{bit}$に圧縮して送る4倍速モードが用意されている.
これらは, `GainSequence::create()`の第1引数で切り替える.
第1引数の型は`GAIN_MODE`であり, 以下の値が用意されている.

* `DUTY_PHASE_FULL` - 位相/振幅を送る, デフォルト
* `PHASE_FULL` - 位相のみを送る
* `PHASE_HALF` - 位相を$\SI{4}{bit}$に圧縮して送る

### Sequence common functions

#### frequency

`Sequence`の周波数を取得する.

#### period_us

`Sequence`の周期を$\SI{}{μs}$単位で取得する.

#### sampling_freq

`Sequence`のサンプリング周波数を取得する.

#### sampling_period_us

`Sequence`のサンプリング周期を$\SI{}{μs}$単位で取得する.

#### sampling_freq_div_ratio

`Sequence`のサンプリング周波数の分周比を取得, 設定する.
サンプリング周波数の基本周波数は$\SI{40}{kHz}$である.
`sampling_freq_div_ratio`は1以上65536以下の整数が指定できる.

```cpp
    seq.sampling_freq_div_ratio() = 5; // 40kHz/5 = 8kHz
```

[^fn_gain_seq]: `PointSequence`のおよそ60倍のレイテンシ.
