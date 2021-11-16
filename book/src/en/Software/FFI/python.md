# python

[pyautd](https://github.com/shinolab/pyautd) provides a wrapper for python 3.7+.

## Installation

It is available at [PyPI](https://pypi.org/project/pyautd3), and you can install it from pip.

```
pip install pyautd3
```

Alternatively, you can install it from the pyautd repository.

```
pip install git+https://github.com/shinolab/pyautd.git
```

### Linux/macOS

If you are using Linux/macOS, you may need administrator privileges. 
In that case, install pyautd with administrative privileges.

```
sudo pip install pyautd3
```

## Usage

Basically, it is designed to be the same as the C++ version.

For example, the equivalent code of [Getting Started](./Users_Manual/getting_started.md) is the following.

```python
from pyautd3 import AUTD, Link, Gain, Modulation, TRANS_SPACING_MM, NUM_TRANS_X, NUM_TRANS_Y


def get_adapter_name():
    adapters = Link.enumerate_adapters()
    for i, adapter in enumerate(adapters):
        print('[' + str(i) + ']: ' + adapter[0] + ', ' + adapter[1])

    index = int(input('choose number: '))
    return adapters[index][0]


if __name__ == '__main__':
    autd = AUTD()

    autd.add_device([0., 0., 0.], [0., 0., 0.])

    ifname = get_adapter_name()
    link = Link.soem(ifname, autd.num_devices())
    if not autd.open(link):
        print(AUTD.last_error())
        exit()

    autd.clear()

    firm_info_list = autd.firmware_info_list()
    for i, firm in enumerate(firm_info_list):
        print(f'[{i}]: CPU: {firm[0]}, FPGA: {firm[1]}')

    autd.silent_mode = True

    x = TRANS_SPACING_MM * ((NUM_TRANS_X - 1) / 2.0)
    y = TRANS_SPACING_MM * ((NUM_TRANS_Y - 1) / 2.0)
    z = 150.0
    g = Gain.focal_point([x, y, z])
    m = Modulation.sine(150)
    autd.send(g, m)

    _ = input()

    autd.close()
```

For a more detailed example, see [pyautd's example](https://github.com/shinolab/pyautd/tree/master/example).

## Trouble shooting

Q. Cannot run from linux or macOS

A. Run as root

```
sudo python
```

If you have any other questions, please send them to [GitHub issues](https://github.com/shinolab/pyautd/issues).
