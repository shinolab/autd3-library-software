# autd3 #

Version: 3.0.2.3

* 旧安定版は[v3.0.0ブランチ](https://github.com/shinolab/autd3-library-software/tree/v3.0.0)にあります.
* 最高1.28MHzの位相更新機能を追加した[version 3.1](https://github.com/shinolab/autd3-library-software/tree/v3.1)もあります.

* 詳細は[Wiki](https://github.com/shinolab/autd3-library-software/wiki)を参照してください.

* 対応するファームウェアは"dist/firmware"内においてあります.

## CAUTION ##

必ず, dist/firmwareにあるv3.0.2のファームウェアを書き込んでから使用してください.

# Change log

* 3.0.2.3 (software)
    * LateralModulationの実装を変更
    * LateralModulationをSpatio-Temporal Modulationにリネーム
    * Linux/Macのtimer.cppのバグを修正

* 3.0.2.2 (software/cpu/fpga)
    * Close時の処理を修正
    * fpgaのmcsファイル圧縮&コンフィグレーション時間短縮
    * cpu/fpgaは互換性あり

* 3.0.2.1 (software)
    * SOEM使用時のバグを修正
    * Rust版追加 (future/autd)

* 3.0.2.0 (software/cpu/fpga)
    * ModulationCalibration実装
        * これにより破壊的な変更がソフト/CPU/FPGAに入った
        * Sync0のデフォルト周期を1msに戻した

* 3.0.1.3
    * AppendGain修正
    * Sync0のデフォルト周期を64msに

* 3.0.1.2
    * mod_reset修正 (100ns)

* 3.0.1.1
    * SOEMのSYNC0同期信号追加
    * mod_reset修正 (100us)

* 3.0.1.0
    * Simple Open EtherCAT Master (SOEM) による制御を追加 
        * これによりLinux, macサポート
        * また, WindowsでもTwinCATを用いなくても良くなった
    * Usb to Ethernet変換ケーブルなども利用可能
    * boostライブラリへの依存を削除

# Author #

Shun Suzuki, 2019
