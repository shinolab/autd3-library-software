# Filter

`Modulation`にデジタルフィルタを作用させるための`Filter`構造体が用意されている.

現在は, FPGA内部のsilent mode実装と等価なLPFのみが用意されている.

## LPF for Silent

```cpp
#include "autd3/modulation/filter/fir.hpp"

...

    const auto m = autd::modulation::Sine::create(150);
    const auto lpf = autd::modulation::filter::FIR::lpf();
    lpf.apply(m);
```

上記のコードで, silent modeをOnにした時にFPGA内部で行われるのと等価なフィルタがModulationデータに作用する.
