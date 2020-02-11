# File: simple.jl
# Project: AUTD3
# Created Date: 11/02/2020
# Author: Shun Suzuki
# -----
# Last Modified: 11/02/2020
# Modified By: Shun Suzuki (suzuki@hapis.k.u-tokyo.ac.jp)
# -----
# Copyright (c) 2020 Hapis Lab. All rights reserved.
# 

using AUTD3

function get_adapter()
    adapters = enumerate_adapters();
    for (i, adapter) in enumerate(adapters)
        println("[" * string(i) * "]: " * adapter[2] * ", " * adapter[1])
    end

    print("Input number: ")
    idx = parse(Int64, readline())

    adapters[idx][2]
end

function main()
    autd = AUTD()

    add_device(autd, (0.f0, 0.f0, 0.f0), (0.f0, 0.f0, 0.f0))
    add_device(autd, (0.f0, 0.f0, 0.f0), (0.f0, 0.f0, 0.f0))

    adapter = get_adapter()

    open_autd(autd, SOEM, adapter)

    g = focal_point_gain((90.f0, 80.f0, 150.f0))
    m = sine_modulation(150)

    append_gain_sync(autd, g)
    append_modulation_sync(autd, m)

    println("press any key to exit...")
    readline();

    dispose(g)
    dispose(m)
    dispose(autd)
end

main();
println("finish")