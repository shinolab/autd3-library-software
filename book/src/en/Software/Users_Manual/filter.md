# Filter

`Filter` class is for applying digital filters to `Modulation`.

Currently, only LPF which is equivalent to _silent mode_ implementation in FPGA is available.

## LPF for Silent

```cpp
#include "autd3/modulation/filter/fir.hpp".

...

    const auto m = autd::modulation::Sine::create(150);
    const auto lpf = autd::modulation::filter::FIR::lpf();
    lpf.apply(m);
```

With the above code, the modulation data is filtered in a way that is equivalent to what is done in the FPGA when _silent mode_ is turned on.
