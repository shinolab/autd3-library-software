# Sequence

SDK provides functions to switch `Gain` periodically.
These functions are classified into `stm` and `Sequence`.
In the `stm`, any `Gain` can be used as long as the memory of the host is available, but the time precision is low because it is executed by a software timer.
The latter is executed by a hardware timer and has high accuracy but has a strong restriction.

## stm

The following is a sample of moving a single focus periodically on the circumference using `stm`.
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
In the above example, 100 points are sampled at equal intervals on a circle of radius $\SI{20}{mm}$ whose center is at `center`.
A focus moves on the circle while leaping from a sampled point to another at a frequency of $\SI{0.5}{Hz}$.

To use `stm`, get a controller for `stm` with `Controller::stm`.
Then, add `Gain` to this `stm` controller with `add_gain`.
And, start the `stm` with the `start` function.
If you want to pause the `stm`, call the `stop` function.
To resume, call `start` again after `stop`.
Finally, call the `finish` function to finish `stm`.

The use of the original `Controller` is prohibited between the acquisition of the `stm` controller and the call to `finish`.

## Sequence

`Sequence` realizes Spatio-Temporal Modulation with a hardware timer.
SDK provides `PointSequence` which supports only a single focus moving, and `GainSequence` which supports arbitrary `Gain` moving.

### PointSequence

`PointSequence` has the following restrictions.

* Maximum sampling point is 65536.
* Sampling frequency is $\SI{40}{kHz}/N, N=1,2,... ,65536$

The usage of `PointSequence` is almost the same as that of `stm`.
```cpp
  autd::sequence::PointSequence seq;

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

Due to the constraints on the number of sampling points and sampling period, the specified frequency and the actual frequency may differ.
For example, in the above example, since $200$ points are sampled and frequency is $\SI{1}{Hz}$, the sampling frequency should be $\SI{200}{Hz}=\SI{40}{kHz}/200$, which satisfies the constraint.
However, for example, if `point_num`=199, the sampling frequency must be $\SI{199}{Hz}$, but there is no $N=1,2,...,65535$ such that $\SI{199}{Hz}=\SI{40}{kHz}/N$ is satisfied. 
Therefore, the nearest $N$ will be selected. 
This causes a shift between the specified frequency and the actual frequency.
The `set_frequency` function returns the actual frequency.

### GainSequence

`GainSequence` can handle any `Gain`, but the number of `Gain`s is reduced to 2048 instead.

The usage of `GainSequence` is almost the same as that of `PointSequence`.
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
The frequency constraint is also the same as for `PointSequence`.

Since `GainSequence` sends all phase/amplitude data, the latency is large[^fn_gain_seq].
To reduce the latency, `GainSequence` provides additional two modes: `PHASE_FULL` mode which sends only phase data, and `PHASE_HALF` mode which sends only phase data compressed to $\SI{4}{bit}$.
These modes are switched by the first argument of `GainSequence::create()`.
The type of the first argument is `GAIN_MODE`, and the following values are provided.

* `DUTY_PHASE_FULL` - Send phase/amplitude, default
* `PHASE_FULL` - Send only phase. This mode has half latency against for `DUTY_PHASE_FULL`
* `PHASE_HALF` - Send only phase compressed to $\SI{4}{bit}$. This mode has quarter latency against for `DUTY_PHASE_FULL`

### Sequence common functions

#### frequency

Get frequency of `Sequence`

#### period_us

Get cycle of `Sequence` in $\SI{}{μs}$

#### sampling_freq

Get sampling frequency of `Sequence`

#### sampling_period_us

Get sampling cycle of `Sequence` in $\SI{}{μs}$

#### sampling_freq_div_ratio

Get and set the sampling frequency division ratio of `Sequence`.
The base frequency of sampling frequency is $\SI{40}{kHz}$.
The `sampling_freq_div_ratio` can be an integer from 1 to 65536.

```cpp
    seq.sampling_freq_div_ratio() = 5; // 40kHz/5 = 8kHz
```

[^fn_gain_seq]: Approximately 60x latency against for `PointSequence`.
