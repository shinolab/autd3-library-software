# autd3 #

旧安定版は[v3.0.0ブランチ](https://github.com/shinolab/autd3-library-software/tree/v3.0.0)にあります.

これは https://github.com/shinolab/autd をフォークし, C#, Unityの機能等を追加したものです.

Installは[Wiki](https://github.com/shinolab/autd3-library-software/wiki/Install-(dev))を参照してください.

以下の変更がなされています
* Simple Open EtherCAT Master (SOEM) による制御を追加 
    * これによりLinux, macサポート
    * また, WindowsでもTwinCATを用いなくても良くなった
    * Usb to Ethernet変換ケーブルなども利用可能
* boostライブラリへの依存を削除

# Author #

Shun Suzuki, 2018-2019
