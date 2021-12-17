# Julia

[AUTD3.jl](https://github.com/shinolab/AUTD3.jl) provides a wrapper for Julia.

## Installation

You can install AUTD3.jl via package manager.

```
(v1.7) pkg> add https://github.com/shinolab/AUTD3.jl.git
```

### Linux/macOS

If you are using Linux/macOS, you may need administrator privileges. 
In that case, install AUTD3.jl with administrative privileges.

## Usage

Basically, it is designed to be the same as the C++ version.

For example, the equivalent code of [Getting Started](./Users_Manual/getting_started.md) is the following.

```julia
using Printf

using AUTD3
using StaticArrays
using StaticArrays:size

function get_adapter()
    adapters = enumerate_adapters();
    for (i, adapter) in enumerate(adapters)
        @printf("[%d]: %s, %s\n", i, adapter[2], adapter[1])
    end

    print("Input number: ")
    idx = tryparse(Int64, readline())
    if idx === nothing || idx > length(adapters) || idx < 1
        println("choose correct number!")
        return ""
    end

    adapters[idx][2]
end

function main()
    autd = AUTD()

    add_device(autd, SVector(0., 0., 0.), SVector(0., 0., 0.))

    adapter = get_adapter()
    link = soem_link(adapter, num_devices(autd))
    if !open_autd(autd, link)
        println(last_error())
        return
    end

    clear(autd)

    firm_info_list = firmware_info_list(autd)
    for (i, firm_info) in enumerate(firm_info_list)
        @printf("AUTD[%d]: CPU: %s, FPGA: %s\n", i, firm_info[1], firm_info[2])
    end

    set_silent_mode(autd, true)

    x = TRANS_SPACING_MM * ((NUM_TRANS_X - 1) / 2.0)
    y = TRANS_SPACING_MM * ((NUM_TRANS_Y - 1) / 2.0)
    z = 150.0
    g = focal_point_gain(SVector(x, y, z))
    freq::Int32 = 150
    m = sine_modulation(freq)

    send(autd, g, m)

    readline()

    dispose(g)
    dispose(m)
    dispose(autd)
end

main()
```

For a more detailed example, see [AUTD3.jl's example](https://github.com/shinolab/AUTD3.jl/tree/master/example).

## Trouble shooting

Q. Cannot run from linux or macOS

A. Run as root

```
sudo julia
```

If you have any other questions, please send them to [GitHub issues](https://github.com/shinolab/AUTD3.jl/issues).
