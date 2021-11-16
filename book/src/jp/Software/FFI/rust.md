# Rust

[rust-autd](https://github.com/shinolab/rust-autd)ではRust版のライブラリを提供している.

Rust版のライブラリはC++版をラップしたものではなく, Rustで再実装している.
そのため, 一部機能が異なる可能性がある.

## Installation

[crate.io](https://crates.io/crates/autd3)で公開しているので, `cargo.toml`にて以下のようにすれば良い.

```
[dependencies]
autd3 = "1.9.1"
```

また, 各種Link, Gain等は別crateとして公開しているので必要に応じてdependenciesに追加すること.
```
[dependencies]
autd3-soem-link = "1.9.0"
autd3-twincat-link = "1.9.0"
autd3-emulator-link = "1.9.0"
autd3-holo-gain = "1.9.0"
```

さらに, 適当な非同期ランタイムが必要になる. 以下の例ではtokioを用いる.
```
[dependencies]
tokio = { version = "1.6.1", features = ["rt", "time", "rt-multi-thread"]}
```

## Usage

基本的には, C++版と同じになるように設計している.

たとえば, [Getting Started](../Users_Manual/getting_started.md)と等価なコードは以下のようになる.

```rust
use autd3::prelude::*;
use autd3_soem_link::{EthernetAdapters, SoemLink};
use std::io::{self, Write};

fn get_adapter() -> String {
    let adapters: EthernetAdapters = Default::default();
    for (index, adapter) in adapters.into_iter().enumerate() {
        println!("[{}]: {}", index, adapter);
    }

    let i: usize;
    let mut s = String::new();
    loop {
        print!("Choose number: ");
        io::stdout().flush().unwrap();

        io::stdin().read_line(&mut s).unwrap();
        match s.trim().parse() {
            Ok(num) if num < adapters.len() => {
                i = num;
                break;
            }
            _ => continue,
        };
    }
    let adapter = &adapters[i];
    adapter.name.to_string()
}

async fn main_task() {
    let mut geometry = Geometry::new();
    geometry.add_device(Vector3::zeros(), Vector3::zeros());

    let ifname = get_adapter();
    let link = SoemLink::new(&ifname, geometry.num_devices() as u16, 1, |msg| {
        eprintln!("unrecoverable error occurred: {}", msg);
        std::process::exit(-1);
    });

    let mut autd = Controller::open(geometry, link).expect("Failed to open");

    autd.clear().await.unwrap();

    println!("***** Firmware information *****");
    let firm_list = autd.firmware_infos().await.unwrap();
    for firm_info in firm_list {
        println!("{}", firm_info);
    }
    println!("********************************");

    autd.silent_mode = true;

    let center = Vector3::new(
        TRANS_SPACING_MM * ((NUM_TRANS_X - 1) as f64 / 2.0),
        TRANS_SPACING_MM * ((NUM_TRANS_Y - 1) as f64 / 2.0),
        150.0,
    );
    let mut g = Focus::new(center);
    let mut m = Sine::new(150);
    autd.send(&mut g, &mut m).await.unwrap();

    let mut _s = String::new();
    io::stdin().read_line(&mut _s).unwrap();

    autd.close().await.unwrap();
}

fn main() {
    let rt = tokio::runtime::Runtime::new().unwrap();
    rt.block_on(async { main_task().await });
}
```

より詳細なサンプルは[rust-autdのexample](https://github.com/shinolab/rust-autd/tree/master/autd3-examples)を参照されたい.

## Trouble shooting

質問があれば[GitHubのissue](https://github.com/shinolab/rust-autd/issues)にてお願いします.
