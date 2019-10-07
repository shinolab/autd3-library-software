# autd3 #

Version: 3.0.1.1

旧安定版は[v3.0.0ブランチ](https://github.com/shinolab/autd3-library-software/tree/v3.0.0)にあります.

詳細は[Wiki](https://github.com/shinolab/autd3-library-software/wiki)を参照してください.

対応するファームウェアは"dist/firmware"内においてあります.

# Change log

* 3.0.1.1
    * SOEMのSYNC0同期信号追加
    * mod_reset修正

* 3.0.1.0
    * Simple Open EtherCAT Master (SOEM) による制御を追加 
        * これによりLinux, macサポート
        * また, WindowsでもTwinCATを用いなくても良くなった
    * Usb to Ethernet変換ケーブルなども利用可能
    * boostライブラリへの依存を削除

# Author #

Shun Suzuki, 2019
