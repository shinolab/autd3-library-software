'''
File: simple.py
Project: python
Created Date: 11/02/2020
Author: Shun Suzuki
-----
Last Modified: 11/02/2020
Modified By: Shun Suzuki (suzuki@hapis.k.u-tokyo.ac.jp)
-----
Copyright (c) 2020 Hapis Lab. All rights reserved.

'''

from pyautd import *
import sys


def get_adapter_name():
    adapters = AUTD.enumerate_adapters()
    for i, adapter in enumerate(adapters):
        print('[' + str(i) + ']: ' + adapter[0] + ', ' + adapter[1])

    index = int(input('choose number: '))
    return adapters[index][0]


if __name__ == '__main__':
    dll_location = './autd3capi.dll'
    init_autd3(dll_location)

    adapter = get_adapter_name()

    autd = AUTD()

    autd.add_device([0, 0, 0], [0, 0, 0])
    autd.add_device([0, 0, 0], [0, 0, 0])

    autd.open(LinkType.SOEM, adapter)

    f = AUTD.focal_point_gain(90, 80, 150)
    m = AUTD.sine_modulation(150)

    autd.append_gain_sync(f)
    autd.append_modulation_sync(m)

    print('enter any key to exit...')
    sys.stdin.readline()

    m = AUTD.modulation(255)
    autd.append_modulation_sync(m)

    f1 = AUTD.focal_point_gain(87.5, 80, 150)
    f2 = AUTD.focal_point_gain(92.5, 80, 150)
    autd.append_stm_gain(f1)
    autd.append_stm_gain(f2)

    autd.start_stm(50)

    print('enter any key to exit...')
    sys.stdin.readline()

    autd.finish_stm()

    autd.dispose()
