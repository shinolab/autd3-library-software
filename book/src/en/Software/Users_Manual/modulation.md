# Modulation

`Modulation` is a mechanism to control Amplitude Modulation.
The `Modulation` is realized by sampling $\SI{8}{bit}$ data stored in a buffer in FPGA at a certain sampling rate and multiplying it by the duty ratio.
Currently, `Modulation` has the following limitations.

* Maximum buffer size is 65536.
* Sampling rate is $\SI{40}{kHz}/N, N=1,2,... ,65536$.
* Modulation is the same for all devices.
* Modulation is automatically looped.

The SDK provides `Modulation`s to generate several kinds of AM by default.

## Static

Without amplitude modulation.

```cpp
  const auto m = autd::modulation::Static::create();
  autd->send(m);
```

Note that the first argument can be a value of `uint8_t` (default is 255), which can be used to change the output of the ultrasound uniformly.

## Sine

`Modulation` for deforming the sound pressure into a sine wave shape.
```cpp
  const auto m = autd::modulation::Sine::create(f, amplitude, offset); 
```

The first argument is the frequency $f$, the second argument is $amplitude$ (1 by default), and the third argument is $offset$ (0.5 by default), and the sound pressure waveform will be
$$
    \frac{amplitude}{2} \times \sin(ft) + offset.
$$
Here, values out of$[0,1]$ are clamped to fit in $[0,1]$.
The sampling frequency is set to $\SI{4}{kHz}$ ($N=10$) by default.

## SinePressure

`Modulation` for deforming the radiation pressure, i.e. the square of the sound pressure, into sine wave shape.
The arguments are the same as for `Sine`.

## SineLegacy

This `Modulation` is compatible with `Sine Modulation` in the old version.
The frequency can be a value of `double`, but it is not exactly the specified frequency, the closest frequency among the possible output frequencies is chosen.
Also, the duty ratio, not the sound pressure, becomes sine wave.

## Square

Square wave-shaped `Modulation`.

```cpp
  const auto m = autd::modulation::Square::create(f, low, high); 
```
The first argument is the frequency $f$, the second argument is low level (0 by default), the third argument is high level (255 by default).
The duty ratio of the square modulation can be specified as the fourth argument.
The duty ratio of the square modulation is defined by $t_\text{high}/T = t_\text{high}f$, where $t_\text{high}$ is the time to output high in a period $T=1/f$.

## Create Custom Modulation Tutorial

You can create your own `Modulation`, as well as `Gain`.
Here, we try to make a `Burst` which outputs only for a moment in a cycle[^fn_burst].

The following is a sample of `Burst`.
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

Like `Gain`, `Modulation` has a `Modulation::calc` method called in `Controller::send`.
So, in this `calc`, you should rewrite the contents of `buffer`.
$N$, which determines the `Modulation` sampling frequency $\SI{40}{kHz}/N$, is specified as the first argument of the constructor of `Modulation`.
In this example, the default value is $N=10$, so the sampling frequency is $\SI{4}{kHz}$.
Furthermore, for example, if $\text{buf\_size}=4000$, the AM value $0$ will be sampled $3999$ times and then $255$ once.
Thus, the AM such that outputs only $\SI{0.25}{ms}$ in the period $\SI{1}{s}$ will be applied.

## Modulation common functions

### sampling_freq_div_ratio

You can use `sampling_freq_div_ratio` function to check and set the sampling frequency division ratio $N$.
The base frequency of sampling frequency is $\SI{40}{kHz}$.
The `sampling_freq_div_ratio` can be any integer between 1 and 65536.

```cpp
    m->sampling_freq_div_ratio() = 5; // 40kHz/5 = 8kHz
```

### sampling_freq

You can get the sampling frequency by `sampling_freq`.

[^fn_burst]: This `Burst` Modulation does not exists in SDK.
