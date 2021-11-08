# python

[pyautd](https://github.com/shinolab/pyautd)はpythonに対応したラッパーを提供している.

## Installation

[PyPI](https://pypi.org/project/pyautd3/)で公開しているので, pipからインストールすること.

```
pip install pyautd3
```

あるいは, pyautdのリポジトリからインストールできる.

```
pip install git+https://github.com/shinolab/pyautd.git
```

### Linux/macOS

Linux/macOSを使用する場合, 管理者権限が必要な場合がある. その時は, 管理者権限付きでインストールすること.

```
sudo pip install pyautd3
```

## Usage

基本的には, C++版と同じになるように設計している.

たとえば, [Getting Started](../Users_Manual/getting_started.md)と等価なコードは以下のようになる.

```python

from pyautd3 import AUTD, Link, Gain, Modulation

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

    g = Gain.focal_point([90, 70, 150])
    m = Modulation.sine(150)
    autd.send(g, m)

    _ = input()

    autd.close()
```

より詳細なサンプルは[pyautdのexample](https://github.com/shinolab/pyautd/tree/master/example)を参照されたい.

## Trouble shooting

Q. linuxやmacから実行できない

A. 管理者権限で実行する

```
sudo python
```

その他, 質問があれば[GitHubのissue](https://github.com/shinolab/pyautd/issues)にてお願いします.
