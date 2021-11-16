# Modulation

`Modulation`はAM変調を制御するための仕組みである.
`Modulation`は, バッファに貯められた$\SI{8}{bit}$データから, 一定のサンプリングレートでデータを順番にサンプリングし, Duty比に掛け合わすことで実現されている.
現在, `Modulation`には以下の制約がある.

* バッファサイズは最大で65536
* サンプリングレートは$\SI{40}{kHz}/N, N=1,2,...,65536$
* Modulationは全デバイスで共通
* Modulationは自動でループする. 1ループだけ, 等の制御は不可能.

SDKにはデフォルトでいくつかの種類のAMを生成するための`Modulation`がデフォルトで用意されている.

## Static

変調なし.

```cpp
  const auto m = autd::modulation::Static::create();
  autd->send(m);
```

なお, 第1引数は`uint8_t`の値を引数に取れ (デフォルトは255), 超音波の出力を一律で変更するために使うことができる.

## Sine

音圧をSin波状に変形するための`Modulation`.
```cpp
  const auto m = autd::modulation::Sine::create(f, amplitude, offset); 
```

第1引数は周波数$f$, 第2引数は$amplitude$ (デフォルトで1), 第3引数は$offset$ (デフォルトで0.5)になっており, 音圧の波形が
$$
    \frac{amplitude}{2} \times \sin(ft) + offset
$$
となるようなAMをかける.
ただし, 上記で$[0,1]$を超えるような値は$[0,1]$に収まるように変換される.
また, サンプリング周波数はデフォルトで$\SI{4}{kHz}$ ($N=10$) になっている.

## SinePressure

放射圧, すなわち, 音圧の二乗をSin波状に変形するための`Modulation`.
引数等は`Sine`と同じ.

## SineLegacy

古いversionにあった`Sine Modulation`と互換.
周波数として, `double`の値を取れるが, 厳密に指定周波数になるのではなく, 出力可能な周波数の内, 最も近い周波数が選ばれる.
また, 音圧ではなくDuty比がSin波状になる.

## Square

矩形波状の`Modulation`.

```cpp
  const auto m = autd::modulation::Square::create(f, low, high); 
```
第1引数は周波数$f$, 第2引数はlow (デフォルトで0), 第3引数はhigh (デフォルトで255)になっており, 音圧の波形はlowとhighが周波数$f$で繰り返される.
また, 第4引数にduty比を指定できる.
duty比は$t_\text{high}/T = t_\text{high}f$で定義される, ここで, $t_\text{high}$は1周期$T=1/f$の内, highを出力する時間である.

## Create Custom Modulation Tutorial

`Modulation`も独自の`Modulation`を作成することができる.
ここでは, 周期中のある一瞬だけ出力する`Burst`を作ってみる[^fn_burst].

以下が, この`Burst`のサンプルである.
```cpp
class Burst final : public autd::core::Modulation {
 public:
  static autd::ModulationPtr create(size_t buf_size = 4000, uint16_t N = 10) {
    return std::make_shared<BurstModulation>(buf_size, N);
  }
  
  void calc() override {
    this->_buffer.resize(_buf_size, 0);
    this->_buffer[_buf_size - 1] = 0xFF;
  }

  Burst(const size_t buf_size, const uint16_t N) : Modulation(N), _buf_size(buf_size) {}

 private:
  size_t _buf_size;
};
```

`Modulation`も`Gain`と同じく, `Controller::send`内部で`Modulation::calc`メソッドが呼ばれる.
この`calc`の中で, `buffer`の中身を書き換えれば良い.
`Modulation`サンプリング周波数$\SI{40}{kHz}/N$を決定する$N$は`Modulation`のコンストラクタの第1引数で指定する.
この例だと, デフォルトで$N=10$なので, サンプリング周波数は$\SI{4}{kHz}$になる.
さらに, 例えば, $\text{buf\_size}=4000$とすると, AMは$0$が$3999$回サンプリングされた後, $255$が一回サンプリングされる.
したがって, 周期$\SI{1}{s}$の中で, $\SI{0.25}{ms}=1/\SI{4}{kHz}$だけ出力されるようなAMがかかる.

## Modulation common functions

### sampling_freq_div_ratio

`sampling_freq_div_ratio`でサンプリング周波数の分周比$N$の確認, 設定ができる.
サンプリング周波数の基本周波数は$\SI{40}{kHz}$である.
`sampling_freq_div_ratio`は1以上65536以下の整数が指定できる.

```cpp
    m->sampling_freq_div_ratio() = 5; // 40kHz/5 = 8kHz
```

### sampling_freq

`sampling_freq`でサンプリング周波数を取得する.

[^fn_burst]: SDKにはない.
