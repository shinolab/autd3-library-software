
## Usage

* SOEM
    ```cargo run --release --example soem```

* TwinCAT
    ```cargo run --release --example twincat```

if you are using Linux/Mac, you may need to run as a superuser. For example,
```
cargo build --release --example soem
sudo ./target/release/examples/soem
```
