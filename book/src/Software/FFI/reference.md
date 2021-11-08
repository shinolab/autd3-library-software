# API Reference

c言語向けのAPIは[client/capi](https://github.com/shinolab/autd3-library-software/tree/master/client/capi)以下で定義されている.
以下に, このAPIのリファレンスを載せる. 
実際の利用方法は, [C#](https://github.com/shinolab/autd3sharp)/[python](https://github.com/shinolab/pyautd)/[Julia](https://github.com/shinolab/AUTD3.jl)のラッパーライブラリを参照されたい.

> Note: なお, 呼び出し規約は特に明示していない. x86の規定はおそらくcdeclになっていると思われるが確認しておらず, x86から使用するとエラーがでるかもしれない.

##  AUTDCreateController (autd3capi)

Controllerを作成する.

作成した`Controller`は最後に`AUTDFreeController`で開放する必要がある.

| Argument name / return       | type             | in/out | description                                                   |
|------------------------------|------------------|--------|--------------------------------------------------------------|
| out                          | void**    | out    | pointer to ControllerPtr                                     |
| return                       | void      | -      | nothing                                                      |

##  AUTDOpenController (autd3capi)

Controllerをopenする. handleは`AUTDCreateController`で作成したものを使う.
linkは各々のlinkの生成関数で作成したものを使う.

この関数は失敗した場合にfalseを返す.
falseの場合には`AUTDGetLastError`でエラーメッセージを取得できる.

| Argument name / return       | type             | in/out | description                                                   |
|------------------------------|------------------|--------|--------------------------------------------------------------|
| handle                       | void*      | in     | ControllerPtr                                                |
| link                         | void*      | in     | LinkPtr                                                      |
| return                       | bool       | -      | true if success                                              |

##  AUTDAddDevice (autd3capi)

ControllerにDeviceを追加する.
handleは`AUTDCreateController`で作成したものを使う. x, y, zは位置で, rz1, ry, rz2はZYZのオイラー角である.

この関数は追加されたDeviceのIdを返す.

| Argument name / return       | type             | in/out | description                                                   |
|------------------------------|------------------|--------|--------------------------------------------------------------|
| handle                       | void*      | in     | ControllerPtr                                                |
| x                            | double     | in     | x coordinate of position in millimeter                       |
| y                            | double     | in     | y coordinate of position in millimeter                       |
| z                            | double     | in     | z coordinate of position in millimeter                       |
| rz1                          | double     | in     | first angle of ZYZ euler angle in radian                     |
| ry                           | double     | in     | second angle of ZYZ euler angle in radian                    |
| rz2                          | double     | in     | third angle of ZYZ euler angle in radian                     |
| gid                          | int32_t    | in     | group Id                                                     |
| return                       | int32_t    | -      | Device Id                                                    |

##  AUTDAddDeviceQuaternion (autd3capi)

ControllerにDeviceを追加する.
handleは`AUTDCreateController`で作成したものを使う. 
x, y, zは位置で, qw, qx, qy, qzは回転を表すクオータニオンである.

この関数は追加されたDeviceのIdを返す.

| Argument name / return       | type             | in/out | description                                                   |
|------------------------------|------------------|--------|--------------------------------------------------------------|
| handle                       | void*      | in     | ControllerPtr                                                |
| x                            | double     | in     | x coordinate of position in millimeter                       |
| y                            | double     | in     | y coordinate of position in millimeter                       |
| z                            | double     | in     | z coordinate of position in millimeter                       |
| qw                           | double     | in     | quaternion of rotation                                       |
| qx                           | double     | in     | quaternion of rotation                                       |
| qy                           | double     | in     | quaternion of rotation                                       |
| qz                           | double     | in     | quaternion of rotation                                       |
| gid                          | int32_t    | in     | group Id                                                     |
| return                       | int32_t    | -      | Device Id                                                    |

##  AUTDDeleteDevice (autd3capi)

Controllerから指定されたインデックスのDeviceを削除する.
handleは`AUTDCreateController`で作成したものを使う.

この関数は削除されたDeviceのIdを返す.

| Argument name / return       | type             | in/out | description                                                   |
|------------------------------|------------------|--------|--------------------------------------------------------------|
| handle                       | void*      | in     | ControllerPtr                                                |
| idx                          | int32_t    | in     | Device index                                                 |
| return                       | int32_t    | -      | Deleted device Id                                            |

##  AUTDClearDevices (autd3capi)

Controllerから全てのDeviceを削除する.
handleは`AUTDCreateController`で作成したものを使う.

| Argument name / return       | type             | in/out | description                                                   |
|------------------------------|------------------|--------|--------------------------------------------------------------|
| handle                       | void*      | in     | ControllerPtr                                                |
| return                       | void       | -      | nothing                                                      |

##  AUTDCloseController (autd3capi)

ControllerをCloseする. handleは`AUTDCreateController`で作成したものを使う.

この関数はエラーが発生した場合に0未満の値を返す.
エラーが生じた場合には`AUTDGetLastError`でエラーメッセージを取得できる.
また, check ackフラグがOn, かつ, 返り値が0より大きい場合は,
データが実際のデバイスで処理されたことを保証する.

| Argument name / return       | type             | in/out | description                                                                              |
|------------------------------|------------------|--------|-----------------------------------------------------------------------------------------|
| handle                       | void*      | in     | ControllerPtr                                                                           |
| return                       | int32_t    | -      | if $>0$, it guarantees devices have processed data. if $<0$, error ocurred.             |

##  AUTDClear (autd3capi)

デバイス内の状態をClearする.
handleは`AUTDCreateController`で作成したものを使う.

この関数はエラーが発生した場合に0未満の値を返す.
エラーが生じた場合には`AUTDGetLastError`でエラーメッセージを取得できる.
また,check ackフラグがOn, かつ, 返り値が0より大きい場合は,
データが実際のデバイスで処理されたことを保証する.

| Argument name / return       | type             | in/out | description                                                                              |
|------------------------------|------------------|--------|-----------------------------------------------------------------------------------------|
| handle                       | void*      | in     | ControllerPtr                                                                           |
| return                       | int32_t    | -      | if $>0$, it guarantees devices have processed data. if $<0$, error ocurred.             |

##  AUTDFreeController (autd3capi)

Controllerを削除する. handleは`AUTDCreateController`で作成したものを使う.
この関数以降handleは使用できない.

| Argument name / return       | type             | in/out | description                                                                              |
|------------------------------|------------------|--------|-----------------------------------------------------------------------------------------|
| handle                       | void*      | in     | ControllerPtr                                                                           |
| return                       | void       | -      | nothing                                                                                 |

##  AUTDIsOpen (autd3capi)

ControllerがOpenされているかどうかを返す.
handleは`AUTDCreateController`で作成したものを使う.

| Argument name / return       | type             | in/out | description                                                                              |
|------------------------------|------------------|--------|-----------------------------------------------------------------------------------------|
| handle                       | void*      | in     | ControllerPtr                                                                           |
| return                       | bool       | -      | true if controller is open                                                              |

##  AUTDGetOutputEnable (autd3capi)

Output enableフラグを返す.
handleは`AUTDCreateController`で作成したものを使う.

| Argument name / return       | type             | in/out | description                                                                              |
|------------------------------|------------------|--------|-----------------------------------------------------------------------------------------|
| handle                       | void*      | in     | ControllerPtr                                                                           |
| return                       | bool       | -      | output enable flag                                                                      |

##  AUTDGetSilentMode (autd3capi)

silent modeフラグを返す. handleは`AUTDCreateController`で作成したものを使う.

| Argument name / return       | type             | in/out | description                                                                              |
|------------------------------|------------------|--------|-----------------------------------------------------------------------------------------|
| handle                       | void*      | in     | ControllerPtr                                                                           |
| return                       | bool       | -      | silent mode flag                                                                        |

##  AUTDGetForceFan (autd3capi)

Force fan flagを返す. handleは`AUTDCreateController`で作成したものを使う.

| Argument name / return       | type             | in/out | description                                                                              |
|------------------------------|------------------|--------|-----------------------------------------------------------------------------------------|
| handle                       | void*      | in     | ControllerPtr                                                                           |
| return                       | bool       | -      | Force fan flag                                                                          |

##  AUTDGetReadsFPGAInfo (autd3capi)

reads FPGA Info flagを返す.
handleは`AUTDCreateController`で作成したものを使う.

| Argument name / return       | type             | in/out | description                                                                              |
|------------------------------|------------------|--------|-----------------------------------------------------------------------------------------|
| handle                       | void*      | in     | ControllerPtr                                                                           |
| return                       | bool       | -      | reads FPGA Info flag                                                                    |

##  AUTDGetOutputBalance (autd3capi)

Output balanceフラグを返す.
handleは`AUTDCreateController`で作成したものを使う.

| Argument name / return       | type             | in/out | description                                                                              |
|------------------------------|------------------|--------|-----------------------------------------------------------------------------------------|
| handle                       | void*      | in     | ControllerPtr                                                                           |
| return                       | bool       | -      | Output balance flag                                                                     |

##  AUTDGetCheckAck (autd3capi)

Check ackフラグを返す. handleは`AUTDCreateController`で作成したものを使う.

| Argument name / return       | type             | in/out | description                                                                              |
|------------------------------|------------------|--------|-----------------------------------------------------------------------------------------|
| handle                       | void*      | in     | ControllerPtr                                                                           |
| return                       | bool       | -      | Check ack flag                                                                          |

##  AUTDSetOutputEnable (autd3capi)

Output enableを設定する.
handleは`AUTDCreateController`で作成したものを使う.

デバイスに実際に反映されるのはsend functionsのどれかを呼び出し後である.

| Argument name / return       | type             | in/out | description                                                                              |
|------------------------------|------------------|--------|-----------------------------------------------------------------------------------------|
| handle                       | void*      | in     | ControllerPtr                                                                           |
| enable                       | bool       | in     | output enable flag                                                                      |
| return                       | void       | -      | nothing                                                                                 |

##  AUTDSetSilentMode (autd3capi)

Silent modeを設定する. handleは`AUTDCreateController`で作成したものを使う.

デバイスに実際に反映されるのはsend functionsのどれかを呼び出し後である.

| Argument name / return       | type             | in/out | description                                                                              |
|------------------------------|------------------|--------|-----------------------------------------------------------------------------------------|
| handle                       | void*      | in     | ControllerPtr                                                                           |
| mode                         | bool       | in     | silent mode flag                                                                        |
| return                       | void       | -      | nothing                                                                                 |

##  AUTDSetReadsFPGAInfo (autd3capi)

FPGAの情報を読み取るかどうかを設定する.
handleは`AUTDCreateController`で作成したものを使う.

デバイスに実際に反映されるのはsend functionsのどれか呼び出し後である.

| Argument name / return       | type             | in/out | description                                                                              |
|------------------------------|------------------|--------|-----------------------------------------------------------------------------------------|
| handle                       | void*      | in     | ControllerPtr                                                                           |
| reads\_fpga\_info            | bool       | in     | read FPGA info flag                                                                     |
| return                       | void       | -      | nothing                                                                                 |

##  AUTDSetForceFan (autd3capi)

Force fan flagを設定する.
handleは`AUTDCreateController`で作成したものを使う.

デバイスに実際に反映されるのはsend functionsのどれかを呼び出し後である.

| Argument name / return       | type             | in/out | description                                                                              |
|------------------------------|------------------|--------|-----------------------------------------------------------------------------------------|
| handle                       | void*      | in     | ControllerPtr                                                                           |
| force                        | bool       | in     | force fan flag                                                                          |
| return                       | void       | -      | nothing                                                                                 |

##  AUTDSetOutputBalance (autd3capi)

Output balance flagを設定する.
handleは`AUTDCreateController`で作成したものを使う.

デバイスに実際に反映されるのはsend functionsのどれかを呼び出し後である.

| Argument name / return       | type             | in/out | description                                                                              |
|------------------------------|------------------|--------|-----------------------------------------------------------------------------------------|
| handle                       | void*      | in     | ControllerPtr                                                                           |
| output_balance               | bool       | in     | Output balance flag                                                                     |
| return                       | void       | -      | nothing                                                                                 |

##  AUTDSetCheckAck (autd3capi)

Check ack flagを設定する.
handleは`AUTDCreateController`で作成したものを使う.

| Argument name / return       | type             | in/out | description                                                                              |
|------------------------------|------------------|--------|-----------------------------------------------------------------------------------------|
| handle                       | void*      | in     | ControllerPtr                                                                           |
| check_ack                    | bool       | in     | Check ack flag                                                                          |
| return                       | void       | -      | nothing                                                                                 |

##  AUTDGetWavelength (autd3capi)

波長を取得する. handleは`AUTDCreateController`で作成したものを使う.

| Argument name / return       | type             | in/out | description                                                                              |
|------------------------------|------------------|--------|-----------------------------------------------------------------------------------------|
| handle                       | void*      | in     | ControllerPtr                                                                           |
| return                       | double     | -      | wavelength                                                                              |

##  AUTDGetAttenuation (autd3capi)

減衰係数を取得する. handleは`AUTDCreateController`で作成したものを使う.

| Argument name / return       | type             | in/out | description                                                                              |
|------------------------------|------------------|--------|-----------------------------------------------------------------------------------------|
| handle                       | void*      | in     | ControllerPtr                                                                           |
| return                       | double     | -      | attenuation coefficient                                                                 |

##  AUTDSetWavelength (autd3capi)

波長を設定する. handleは`AUTDCreateController`で作成したものを使う.

| Argument name / return       | type             | in/out | description                                                                              |
|------------------------------|------------------|--------|-----------------------------------------------------------------------------------------|
| handle                       | void*      | in     | ControllerPtr                                                                           |
| wavelength                   | double     | in     | wavelength                                                                              |
| return                       | void       | -      | nothing                                                                                 |

##  AUTDSetAttenuation (autd3capi)

減衰係数を設定する. handleは`AUTDCreateController`で作成したものを使う.

| Argument name / return       | type             | in/out | description                                                                              |
|------------------------------|------------------|--------|-----------------------------------------------------------------------------------------|
| handle                       | void*      | in     | ControllerPtr                                                                           |
| attenuation                  | double     | in     | attenuation coefficient                                                                 |
| return                       | void       | -      | nothing                                                                                 |

##  AUTDGetFPGAInfo (autd3capi)

FPGAの情報を取得する. handleは`AUTDCreateController`で作成したものを使う.
outポインタが指す領域は, 接続しているデバイスと同じ長さである必要がある.
なお, FPGAの情報は下位1bitがFanが起動しているかどうかを表し,
他のbitは全て0である.

この関数を呼び出す前に`AUTDGetReadsFPGAInfo`でread FPGA info
flagをOnにしておく必要がある.

この関数は失敗した場合にfalseを返す.
falseの場合には`AUTDGetLastError`でエラーメッセージを取得できる.

| Argument name / return       | type             | in/out | description                                                                              |
|------------------------------|------------------|--------|-----------------------------------------------------------------------------------------|
| handle                       | void*      | in     | ControllerPtr                                                                           |
| out                          | uint8_t *  | out    | FPGA informations                                                                        |
| return                       | bool       | -      | true if success                                                                         |

##  AUTDUpdateCtrlFlags (autd3capi)

Control flagを更新する. send functionの一つ. output enable, silent mode, force
fan, reads FPGA info, output balance flagsを設定した後に呼び出すと,
これらの変更が実際に反映される.
handleは`AUTDCreateController`で作成したものを使う.

この関数はエラーが発生した場合に0未満の値を返す.
エラーが生じた場合には`AUTDGetLastError`でエラーメッセージを取得できる.
また,check ackフラグがOn, かつ, 返り値が0より大きい場合は,
データが実際のデバイスで処理されたことを保証する.

| Argument name / return       | type             | in/out | description                                                                              |
|------------------------------|------------------|--------|-----------------------------------------------------------------------------------------|
| handle                       | void*      | in     | ControllerPtr                                                                           |
| return                       | int32_t    | -      | if $>0$, it guarantees devices have processed data. if $<0$, error ocurred.             |

##  AUTDSetOutputDelay (autd3capi)

Output delayを設定する. send functionの.
handleは`AUTDCreateController`で作成したものを使う.
delayは(Deviceの台数)$\times 249$の長さのデータへのポインタである必要がある.

この関数はエラーが発生した場合に0未満の値を返す.
エラーが生じた場合には`AUTDGetLastError`でエラーメッセージを取得できる.
また, check ackフラグがOn, かつ, 返り値が0より大きい場合は,
データが実際のデバイスで処理されたことを保証する.

| Argument name / return       | type             | in/out | description                                                                              |
|------------------------------|------------------|--------|-----------------------------------------------------------------------------------------|
| handle                       | void*      | in     | ControllerPtr                                                                           |
| delay                        | uint8_t *  | in     | pointer to delay data                                                                   |
| return                       | int32_t    | -      | if $>0$, it guarantees devices have processed data. if $<0$, error ocurred.             |

##  AUTDSetDutyOffset (autd3capi)

Duty offsetを設定する. send functionの一つ.
handleは`AUTDCreateController`で作成したものを使う.
offsetは(Deviceの台数)$\times 249$の長さのデータへのポインタである必要がある.

この関数はエラーが発生した場合に0未満の値を返す.
エラーが生じた場合には`AUTDGetLastError`でエラーメッセージを取得できる.
また,check ackフラグがOn, かつ, 返り値が0より大きい場合は,
データが実際のデバイスで処理されたことを保証する.

| Argument name / return       | type             | in/out | description                                                                              |
|------------------------------|------------------|--------|-----------------------------------------------------------------------------------------|
| handle                       | void*      | in     | ControllerPtr                                                                           |
| offset                       | uint8_t *  | in     | pointer to duty offset data                                                             |
| return                       | int32_t    | -      | if $>0$, it guarantees devices have processed data. if $<0$, error ocurred.             |

##  AUTDSetDelayOffset (autd3capi)

Output delayとDuty offsetを設定する. send functionの一つ..
handleは`AUTDCreateController`で作成したものを使う. delay,
offsetは(Deviceの台数)$\times 249$の長さのデータへのポインタである必要がある.

この関数はエラーが発生した場合に0未満の値を返す.
エラーが生じた場合には`AUTDGetLastError`でエラーメッセージを取得できる.
また,check ackフラグがOn, かつ, 返り値が0より大きい場合は,
データが実際のデバイスで処理されたことを保証する.

| Argument name / return       | type             | in/out | description                                                                              |
|------------------------------|------------------|--------|-----------------------------------------------------------------------------------------|
| handle                       | void*      | in     | ControllerPtr                                                                           |
| delay                        | uint8_t *  | in     | pointer to output delay data                                                            |
| offset                       | uint8_t *  | in     | pointer to duty offset data                                                             |
| return                       | int32_t    | -      | if $>0$, it guarantees devices have processed data. if $<0$, error ocurred.             |

##  AUTDGetLastError (autd3capi)

最後に発生したエラーメッセージを取得する.

引数にはエラーメッセージへのポインタを渡す,
このポインタにエラーメッセージがコピーされる. ただし,
引数がnullptrの場合はコピーは行われない. この関数は,
null終端込みのエラーメッセージのサイズを返す.

エラーメッセージの長さは可変なので十分に大きな領域を確保しておくか,
または, errorにnullptrを渡し必要なサイズを取得して再び呼び出すこと.

| Argument name / return       | type             | in/out | description                                                                              |
|------------------------------|------------------|--------|-----------------------------------------------------------------------------------------|
| error                        | char*            | out    | pointer to error message                                                                |
| return                       | int32_t    | -      | length of error message including null terminator                                       |

##  AUTDNumDevices (autd3capi)

接続されているDeviceの数を取得する.
handleは`AUTDCreateController`で作成したものを使う.

| Argument name / return       | type             | in/out | description                                                                              |
|------------------------------|------------------|--------|-----------------------------------------------------------------------------------------|
| handle                       | void*      | in     | ControllerPtr                                                                           |
| return                       | int32_t    | -      | number of devices                                                                       |

##  AUTDNumTransducers (autd3capi)

振動子の総数を取得する.
handleは`AUTDCreateController`で作成したものを使う.

| Argument name / return       | type             | in/out | description                                                                              |
|------------------------------|------------------|--------|-----------------------------------------------------------------------------------------|
| handle                       | void*      | in     | ControllerPtr                                                                           |
| return                       | int32_t    | -      | number of transducers                                                                   |

##  AUTDDeviceIdxForTransIdx (autd3capi)

グローバルな振動子のインデックスをDeviceのインデックスに変換する.
handleは`AUTDCreateController`で作成したものを使う.

| Argument name / return       | type             | in/out | description                                                                              |
|------------------------------|------------------|--------|-----------------------------------------------------------------------------------------|
| handle                       | void*      | in     | ControllerPtr                                                                           |
| global_trans_idx       | int32_t    | in     | global transducer index                                                                 |
| return                       | int32_t    | -      | device index                                                                            |

##  AUTDTransPositionByGlobal (autd3capi)

指定した振動子の位置を取得する.
振動子の指定はグローバルなインデックスでおこなう.
handleは`AUTDCreateController`で作成したものを使う.

| Argument name / return       | type             | in/out | description                                                                              |
|------------------------------|------------------|--------|-----------------------------------------------------------------------------------------|
| handle                       | void*      | in     | ControllerPtr                                                                           |
| global_trans_idx       | int32_t    | in     | global transducer index                                                                 |
| x                            | double*          | out    | x coordinate of transducer position                                                     |
| y                            | double*          | out    | y coordinate of transducer position                                                     |
| z                            | double*          | out    | z coordinate of transducer position                                                     |
| return                       | void       | -      | nothing                                                                                 |


##  AUTDTransPositionByLocal (autd3capi)

指定した振動子の位置を取得する.
振動子の指定はデバイスのインデックスとローカルの振動子インデックスでおこなう.
handleは`AUTDCreateController`で作成したものを使う.

| Argument name / return       | type             | in/out | description                                                                              |
|------------------------------|------------------|--------|-----------------------------------------------------------------------------------------|
| handle                       | void*      | in     | ControllerPtr                                                                           |
| device_idx             | int32_t    | in     | device index                                                                            |
| local_trans_idx        | int32_t    | in     | local transducer index                                                                  |
| x                            | double*          | out    | x coordinate of transducer position                                                     |
| y                            | double*          | out    | y coordinate of transducer position                                                     |
| z                            | double*          | out    | z coordinate of transducer position                                                     |
| retunrn                      | void       | -      | nothing                                                                                 |

##  AUTDDeviceXDirection (autd3capi)

指定したデバイスのx軸方向を取得する.
handleは`AUTDCreateController`で作成したものを使う.

| Argument name / return       | type             | in/out | description                                                                              |
|------------------------------|------------------|--------|-----------------------------------------------------------------------------------------|
| handle                       | void*      | in     | ControllerPtr                                                                           |
| device_idx             | int32_t    | in     | device index                                                                            |
| x                            | double*          | out    | x coordinate of device x-direction                                                      |
| y                            | double*          | out    | y coordinate of device x-direction                                                      |
| z                            | double*          | out    | z coordinate of device x-direction                                                      |
| retunrn                      | void       | -      | nothing                                                                                 |

##  AUTDDeviceYDirection (autd3capi)

指定したデバイスのy軸方向を取得する.
handleは`AUTDCreateController`で作成したものを使う.

| Argument name / return       | type             | in/out | description                                                                              |
|------------------------------|------------------|--------|-----------------------------------------------------------------------------------------|
| handle                       | void*      | in     | ControllerPtr                                                                           |
| device_idx             | int32_t    | in     | device index                                                                            |
| x                            | double*          | out    | x coordinate of device y-direction                                                      |
| y                            | double*          | out    | y coordinate of device y-direction                                                      |
| z                            | double*          | out    | z coordinate of device y-direction                                                      |
| retunrn                      | void       | -      | nothing                                                                                 |

##  AUTDDeviceZDirection (autd3capi)

指定したデバイスのz軸方向を取得する.
handleは`AUTDCreateController`で作成したものを使う.

| Argument name / return       | type             | in/out | description                                                                              |
|------------------------------|------------------|--------|-----------------------------------------------------------------------------------------|
| handle                       | void*      | in     | ControllerPtr                                                                           |
| device_idx             | int32_t    | in     | device index                                                                            |
| x                            | double*          | out    | x coordinate of device z-direction                                                      |
| y                            | double*          | out    | y coordinate of device z-direction                                                      |
| z                            | double*          | out    | z coordinate of device z-direction                                                      |
| retunrn                      | void       | -      | nothing                                                                                 |

##  AUTDGetFirmwareInfoListPointer (autd3capi)

Firmware information listへのポインタを取得する.
handleは`AUTDCreateController`で作成したものを使う.
この関数で作成したlistは最後に`AUTDFreeFirmwareInfoListPointer`で開放する必要がある.

実際のFirmware informationは`AUTDGetFirmwareInfo`で取得する.

この関数はエラーが発生した場合に0未満の値を返す.
エラーが生じた場合には`AUTDGetLastError`でエラーメッセージを取得できる

| Argument name / return       | type             | in/out | description                                                                              |
|------------------------------|------------------|--------|-----------------------------------------------------------------------------------------|
| handle                       | void*      | in     | ControllerPtr                                                                           |
| out                          | void**     | out    | pointer to pointer to Firmware information list                                         |

##  AUTDGetFirmwareInfo (autd3capi)

Firmware informationを取得する.
`p_firm_info_list`は`AUTDGetFirmwareInfoListPointer`で作成したものを使う.

`cpu_ver`, `fpga_ver`は長さ128のバッファを渡せば十分である.

| Argument name / return       | type             | in/out | description                                                                              |
|------------------------------|------------------|--------|-----------------------------------------------------------------------------------------|
| p_firm_info_list       | void*      | in     | pointer to Firmware information list                                                    |
| index                        | int32_t    | in     | Firmware information index                                                              |
| cpu_ver                | char*            | out    | pointer to CPU version string                                                           |
| fpga_ver               | char*            | out    | pointer to FPGA version string                                                          |
| return                       | void       | -      | nothing                                                                                 |

##  AUTDFreeFirmwareInfoListPointer (autd3capi)

`AUTDGetFirmwareInfoListPointer`で取得したFirmware information
listを開放する.

| Argument name / return       | type             | in/out | description                                                                              |
|------------------------------|------------------|--------|-----------------------------------------------------------------------------------------|
| p_firm_info_list       | void*      | in     | pointer to Firmware information list                                                    |
| return                       | void       | -      | nothing                                                                                 |

##  AUTDGainNull (autd3capi)

Gain::Nullを作成する.

作成したGainは最後に`AUTDDeleteGain`で削除する必要がある.

| Argument name / return       | type             | in/out | description                                                                              |
|------------------------------|------------------|--------|-----------------------------------------------------------------------------------------|
| gain                   | void**     | out    | pointer to GainPtr                                                                      |
| return                       | void       | -      | nothing                                                                                 |

##  AUTDGainGrouped (autd3capi)

Gain::Groupedを作成する.

作成したGainは最後に`AUTDDeleteGain`で削除する必要がある.

| Argument name / return       | type             | in/out | description                                                                              |
|------------------------------|------------------|--------|-----------------------------------------------------------------------------------------|
| gain                   | void**     | out    | pointer to GainPtr                                                                      |
| return                       | void       | -      | nothing                                                                                 |

##  AUTDGainGroupedAdd (autd3capi)

Gain::GroupedにGainを追加する.
`grouped_gain`は`AUTDGainGrouped`で作成したものを使う. 
また, idには`AUTDAddDevice`, または, `AUTDAddDeviceQuaternion`で指定したGroup
Idを使用する.

| Argument name / return       | type             | in/out | description                                                                              |
|------------------------------|------------------|--------|-----------------------------------------------------------------------------------------|
| grouped_gain           | void*      | in     | pointer to Grouped Gain                                                                 |
| id                           | int32_t    | in     | Groupe Id                                                                               |
| gain                   | void*      | in     | GainPtr                                                                                 |
| return                       | void       | -      | nothing                                                                                 |

##  AUTDGainFocalPoint (autd3capi)

Gain::FocalPointを作成する.

作成したGainは最後に`AUTDDeleteGain`で削除する必要がある.

| Argument name / return       | type             | in/out | description                                                                              |
|------------------------------|------------------|--------|-----------------------------------------------------------------------------------------|
| gain                   | void**     | out    | pointer to GainPtr                                                                      |
| x                            | double     | in     | x coordinate of focal point                                                             |
| y                            | double     | in     | y coordinate of focal point                                                             |
| z                            | double     | in     | z coordinate of focal point                                                             |
| duty                         | uint8_t    | in     | duty ratio of Gain                                                                      |
| return                       | void       | -      | nothing                                                                                 |

##  AUTDGainBesselBeam (autd3capi)

Gain::Besselを作成する.

作成したGainは最後に`AUTDDeleteGain`で削除する必要がある.

| Argument name / return       | type             | in/out | description                                                                              |
|------------------------------|------------------|--------|-----------------------------------------------------------------------------------------|
| gain                   | void**     | out    | pointer to GainPtr                                                                      |
| x                            | double     | in     | x coordinate of apex                                                                    |
| y                            | double     | in     | y coordinate of apex                                                                    |
| z                            | double     | in     | z coordinate of apex                                                                    |
| n_x                    | double     | in     | x coordinate of direction                                                               |
| n_y                    | double     | in     | y coordinate of direction                                                               |
| n_z                    | double     | in     | z coordinate of direction                                                               |
| theta_z                | double     | in     | angle between the side of the cone and the plane perpendicular to direction of the beam |
| duty                         | uint8_t    | in     | duty ratio of Gain                                                                      |
| return                       | void       | -      | nothing                                                                                 |

##  AUTDGainPlaneWave (autd3capi)

Gain::PlaneWaveを作成する.

作成したGainは最後に`AUTDDeleteGain`で削除する必要がある.

| Argument name / return       | type             | in/out | description                                                                              |
|------------------------------|------------------|--------|-----------------------------------------------------------------------------------------|
| gain                   | void**     | out    | pointer to GainPtr                                                                      |
| n_x                    | double     | in     | x coordinate of direction                                                               |
| n_y                    | double     | in     | y coordinate of direction                                                               |
| n_z                    | double     | in     | z coordinate of direction                                                               |
| duty                         | uint8_t    | in     | duty ratio of Gain                                                                      |
| return                       | void       | -      | nothing                                                                                 |

##  AUTDGainCustom (autd3capi)

Gain::Customを作成する.

作成したGainは最後に`AUTDDeleteGain`で削除する必要がある.

| Argument name / return       | type             | in/out | description                                                                              |
|------------------------------|------------------|--------|-----------------------------------------------------------------------------------------|
| gain                   | void**     | out    | pointer to GainPtr                                                                      |
| data                   | uint16_t*  | in     | pointer to data                                                                         |
| data_length            | int32_t    | in     | length of data                                                                          |
| return                       | void       | -      | nothing                                                                                 |

##  AUTDGainTransducerTest (autd3capi)

Gain::TransducerTestを作成する.

作成したGainは最後に`AUTDDeleteGain`で削除する必要がある.

| Argument name / return       | type             | in/out | description                                                                              |
|------------------------------|------------------|--------|-----------------------------------------------------------------------------------------|
| gain                   | void**     | out    | pointer to GainPtr                                                                      |
| idx                    | int32_t    | in     | global index of transducer                                                              |
| duty                   | uint8_t    | in     | duty ratio                                                                              |
| phase                  | uint8_t    | in     | phase                                                                                   |
| return                       | void       | -      | nothing                                                                                 |

##  AUTDDeleteGain (autd3capi)

作成したGainを削除する.

| Argument name / return       | type             | in/out | description                                                                              |
|------------------------------|------------------|--------|-----------------------------------------------------------------------------------------|
| gain                   | void*      | in     | GainPtr                                                                                 |
| return                       | void       | -      | nothing                                                                                 |

##  AUTDModulationStatic (autd3capi)

Modulation::Staticを作成する.

作成したModulationは最後に`AUTDDeleteModulation`で削除する必要がある.

| Argument name / return       | type             | in/out | description                                                                              |
|------------------------------|------------------|--------|-----------------------------------------------------------------------------------------|
| mod                    | void**     | out    | ModulationPtr                                                                           |
| duty                   | uint8_t    | in     | duty ratio                                                                              |
| return                       | void       | -      | nothing                                                                                 |

##  AUTDModulationCustom (autd3capi)

Modulation::Customを作成する.

作成したModulationは最後に`AUTDDeleteModulation`で削除する必要がある.

| Argument name / return       | type             | in/out | description                                                                              |
|------------------------------|------------------|--------|-----------------------------------------------------------------------------------------|
| mod                    | void**     | out    | ModulationPtr                                                                           |
| buf                    | uint8_t*   | in     | ModulationPtr buffer                                                                    |
| size                   | uint32_t   | in     | length to modulation buffer                                                             |
| return                       | void       | -      | nothing                                                                                 |

##  AUTDModulationSine (autd3capi)

Modulation::Sineを作成する.

作成したModulationは最後に`AUTDDeleteModulation`で削除する必要がある.

| Argument name / return       | type             | in/out | description                                                                              |
|------------------------------|------------------|--------|-----------------------------------------------------------------------------------------|
| mod                    | void**     | out    | ModulationPtr                                                                           |
| freq                   | int32_t    | in     | frequency                                                                               |
| amplitude              | double     | in     | amplitude of sin wave                                                                   |
| offset                 | double     | in     | offset of sin wave                                                                      |
| return                       | void       | -      | nothing                                                                                 |

##  AUTDModulationSinePressure (autd3capi)

Modulation::SinePressureを作成する.

作成したModulationは最後に`AUTDDeleteModulation`で削除する必要がある.

| Argument name / return       | type             | in/out | description                                                                              |
|------------------------------|------------------|--------|-----------------------------------------------------------------------------------------|
| mod                    | void**     | out    | ModulationPtr                                                                           |
| freq                   | int32_t    | in     | frequency                                                                               |
| amplitude              | double     | in     | amplitude of sin wave                                                                   |
| offset                 | double     | in     | offset of sin wave                                                                      |
| return                       | void       | -      | nothing                                                                                 |

##  AUTDModulationSineLegacy (autd3capi)

Modulation::SineLegacyを作成する.

作成したModulationは最後に`AUTDDeleteModulation`で削除する必要がある.

| Argument name / return       | type             | in/out | description                                                                              |
|------------------------------|------------------|--------|-----------------------------------------------------------------------------------------|
| mod                    | void**     | out    | ModulationPtr                                                                           |
| freq                   | double     | in     | frequency                                                                               |
| amplitude              | double     | in     | amplitude of sin wave                                                                   |
| offset                 | double     | in     | offset of sin wave                                                                      |
| return                       | void       | -      | nothing                                                                                 |

##  AUTDModulationSquare (autd3capi)

Modulation::Squareを作成する.

作成したModulationは最後に`AUTDDeleteModulation`で削除する必要がある.

| Argument name / return       | type             | in/out | description                                                                              |
|------------------------------|------------------|--------|-----------------------------------------------------------------------------------------|
| mod                    | void**     | out    | ModulationPtr                                                                           |
| freq                   | int32_t    | in     | frequency                                                                               |
| low                    | uint8_t    | in     | duty ratio at low level                                                                 |
| high                   | uint8_t    | in     | duty ratio at high level                                                                |
| duty                   | double     | in     | duty ratio of the square wave, i.e., ratio of duration of high level to period          |
| return                       | void       | -      | nothing                                                                                 |


##  AUTDModulationSamplingFreqDiv (autd3capi)

Modulationのサンプリング周波数分周比を取得する.

| Argument name / return       | type             | in/out | description                                                                              |
|------------------------------|------------------|--------|-----------------------------------------------------------------------------------------|
| mod                    | void*      | in     | ModulationPtr                                                                           |
| return                       | uint32_t   | -      | sampling frequency division ratio                                                       |

##  AUTDModulationSetSamplingFreqDiv (autd3capi)

Modulationのサンプリング周波数分周比を設定する.

| Argument name / return       | type             | in/out | description                                                                              |
|------------------------------|------------------|--------|-----------------------------------------------------------------------------------------|
| mod                    | void*      | in     | ModulationPtr                                                                           |
| freq_div               | uint32_t   | in     | sampling frequency division ratio                                                       |
| return                       | void       | -      | nothing                                                                                 |

##  AUTDModulationSamplingFreq (autd3capi)

Modulationのサンプリング周波数を取得する.

| Argument name / return       | type             | in/out | description                                                                              |
|------------------------------|------------------|--------|-----------------------------------------------------------------------------------------|
| mod                    | void*      | in     | ModulationPtr                                                                           |
| return                       | double     | -      | sampling frequency                                                                      |

##  AUTDDeleteModulation (autd3capi)

Modulationを削除する.

| Argument name / return       | type             | in/out | description                                                                              |
|------------------------------|------------------|--------|-----------------------------------------------------------------------------------------|
| mod                    | void*      | in     | ModulationPtr                                                                           |
| return                       | void       | -      | nothing                                                                                 |

##  AUTDSequence (autd3capi)

PointSequenceを作成する.
作成したPointSequenceは最後に`AUTDDeleteSequence`で削除する必要がある.

| Argument name / return       | type             | in/out | description                                                                              |
|------------------------------|------------------|--------|-----------------------------------------------------------------------------------------|
| out                    | void**     | out    | pointer to PointSequencePtr                                                             |
| return                       | void       | -      | nothing                                                                                 |

##  AUTDGainSequence (autd3capi)

GainSequenceを作成する.

作成したGainSequenceは最後に`AUTDDeleteSequence`で削除する必要がある.

| Argument name / return       | type             | in/out | description                                                                              |
|------------------------------|------------------|--------|-----------------------------------------------------------------------------------------|
| out                    | void**     | out    | pointer to GainSequencePtr                                                              |
| gain_mode              | uint16_t   | in     | gain mode of GainSequence                                                               |
| return                       | void       | -      | nothing                                                                                 |

##  AUTDSequenceAddPoint (autd3capi)

PointSequenceに制御点を追加する.

`seq`には`AUTDSequence`で作成したPointSequenceを使用する.

この関数は失敗した場合にfalseを返す.
falseの場合には`AUTDGetLastError`でエラーメッセージを取得できる.

| Argument name / return       | type             | in/out | description                                                                              |
|------------------------------|------------------|--------|-----------------------------------------------------------------------------------------|
| seq                    | void*      | in     | GainSequencePtr                                                                         |
| x                      | double     | in     | x coordinate of point                                                                   |
| y                      | double     | in     | y coordinate of point                                                                   |
| z                      | double     | in     | z coordinate of point                                                                   |
| duty                   | uint8_t    | in     | duty ratio of point                                                                     |
| return                       | bool       | -      | true if success                                                                         |

##  AUTDSequenceAddPoints (autd3capi)

PointSequenceに複数の制御点を追加する.

`seq`には`AUTDSequence`で作成したPointSequenceを使用する.

`points`は`points_size`$\times 3$の長さの配列のポインタであり,
x\[0\], y\[0\], z\[0\], x\[1\], y\[1\],
z\[1\],\...のような順番で指定する.
`duties_size`が`points_size`未満の場合,
不足分はduty=0xFFのデータが埋められる.
`duties_size`が`points_size`より大きい場合, 過剰分は無視される.

この関数は失敗した場合にfalseを返す.
falseの場合には`AUTDGetLastError`でエラーメッセージを取得できる.

| Argument name / return       | type             | in/out | description                                                                              |
|------------------------------|------------------|--------|-----------------------------------------------------------------------------------------|
| seq                    | void*      | in     | GainSequencePtr                                                                         |
| points                 | double*    | in     | pointer to points array                                                                 |
| points_size            | uint64_t   | in     | length of points array                                                                  |
| duties                 | double*    | in     | pointer to duties array                                                                 |
| duties_size            | uint64_t   | in     | length of duties array                                                                  |
| return                       | bool       | -      | true if success                                                                         |

##  AUTDSequenceAddGain (autd3capi)

GainSequenceにGainを追加する.

`seq`には`AUTDGainSequence`で作成したGainSequenceを使用する.

この関数は失敗した場合にfalseを返す.
falseの場合には`AUTDGetLastError`でエラーメッセージを取得できる.

| Argument name / return       | type             | in/out | description                                                                              |
|------------------------------|------------------|--------|-----------------------------------------------------------------------------------------|
| seq                    | void*      | in     | GainSequencePtr                                                                         |
| gain                   | void*      | in     | GainPtr                                                                                 |
| return                       | bool       | -      | true if success                                                                         |

##  AUTDSequenceSetFreq (autd3capi)

Sequenceに周波数を設定する.

`seq`には`AUTDSequence`, または, `AUTDGainSequence`で作成したSequenceを使用する.

この関数は実際の周波数を返す.

| Argument name / return       | type             | in/out | description                                                                              |
|------------------------------|------------------|--------|-----------------------------------------------------------------------------------------|
| seq                    | void*      | in     | SequencePtr                                                                             |
| freq                   | double     | in     | frequency                                                                               |
| return                       | double     | -      | actual frequency                                                                        |

##  AUTDSequenceFreq (autd3capi)

Sequenceの周波数を取得する.

`seq`には`AUTDSequence`, または, `AUTDGainSequence`で作成したSequenceを使用する.

| Argument name / return       | type             | in/out | description                                                                              |
|------------------------------|------------------|--------|-----------------------------------------------------------------------------------------|
| seq                    | void*      | in     | SequencePtr                                                                             |
| return                       | double     | -      | actual frequency                                                                        |

##  AUTDSequencePeriod (autd3capi)

Sequenceのμs単位の周期を取得する.

`seq`には`AUTDSequence`, または, `AUTDGainSequence`で作成したSequenceを使用する.

| Argument name / return       | type             | in/out | description                                                                              |
|------------------------------|------------------|--------|-----------------------------------------------------------------------------------------|
| seq                    | void*      | in     | SequencePtr                                                                             |
| return                       | uint32_t   | -      | period in μs                                                   |

##  AUTDSequenceSamplingPeriod (autd3capi)

Sequenceのμs単位のサンプリング周期を取得する.

`seq`には`AUTDSequence`, または, `AUTDGainSequence`で作成したSequenceを使用する.

| Argument name / return       | type             | in/out | description                                                                              |
|------------------------------|------------------|--------|-----------------------------------------------------------------------------------------|
| seq                    | void*      | in     | SequencePtr                                                                             |
| return                       | uint32_t   | -      | sampling period in μs                                          |

##  AUTDSequenceSamplingFreq (autd3capi)

Sequenceのサンプリング周波数を取得する.

`seq`には`AUTDSequence`, または, `AUTDGainSequence`で作成したSequenceを使用する.

| Argument name / return       | type             | in/out | description                                                                              |
|------------------------------|------------------|--------|-----------------------------------------------------------------------------------------|
| seq                    | void*      | in     | SequencePtr                                                                             |
| return                       | double     | -      | sampling freqyency                                                                      |

##  AUTDSequenceSamplingFreqDiv (autd3capi)

Sequenceのサンプリング周波数分周比を取得する.

`seq`には`AUTDSequence`, または, `AUTDGainSequence`で作成したSequenceを使用する.

| Argument name / return       | type             | in/out | description                                                                              |
|------------------------------|------------------|--------|-----------------------------------------------------------------------------------------|
| seq                    | void*      | in     | SequencePtr                                                                             |
| return                       | uint32_t   | -      | sampling freqyency division ratio                                                       |

##  AUTDSequenceSetSamplingFreqDiv (autd3capi)

Sequenceのサンプリング周波数分周比を設定する.

`seq`には`AUTDSequence`, または, `AUTDGainSequence`で作成したSequenceを使用する.

| Argument name / return       | type             | in/out | description                                                                              |
|------------------------------|------------------|--------|-----------------------------------------------------------------------------------------|
| seq                    | void*      | in     | SequencePtr                                                                             |
| freq_div               | uint32_t   | in     | sampling freqyency division ratio                                                       |
| return                       | void       | -      | nothing                                                                                 |

##  AUTDCircumSequence (autd3capi)

円周状のPointSequenceを作成する.

作成したPointSequenceは最後に`AUTDDeleteSequence`で削除する必要がある.

| Argument name / return       | type             | in/out | description                                                                              |
|------------------------------|------------------|--------|-----------------------------------------------------------------------------------------|
| out                    | void**     | out    | pointer to PointSequencePtr                                                             |
| x                      | double     | in     | x coordinate of circumference                                                           |
| y                      | double     | in     | y coordinate of circumference                                                           |
| z                      | double     | in     | z coordinate of circumference                                                           |
| nx                     | double     | in     | x coordinate of normal of circumference                                                 |
| ny                     | double     | in     | y coordinate of normal of circumference                                                 |
| nz                     | double     | in     | z coordinate of normal of circumference                                                 |
| radius                 | double     | in     | radius of circumference                                                                 |
| n                      | uint64_t   | in     | number of sampling points                                                               |
| return                       | void       | -      | nothing                                                                                 |

##  AUTDDeleteSequence (autd3capi)

作成したSequenceを削除する.

| Argument name / return       | type             | in/out | description                                                                              |
|------------------------------|------------------|--------|-----------------------------------------------------------------------------------------|
| seq                    | void*      | in     | SequencePtr                                                                             |
| return                       | void       | -      | nothing                                                                                 |

##  AUTDStop (autd3capi)

出力を停止する. send functionの一つ.

`handle`には`AUTDCreateController`で作成したControllerを使用する.

この関数はエラーが発生した場合に0未満の値を返す.
エラーが生じた場合には`AUTDGetLastError`でエラーメッセージを取得できる.
また,check ackフラグがOn, かつ, 返り値が0より大きい場合は,
データが実際のデバイスで処理されたことを保証する.

| Argument name / return       | type             | in/out | description                                                                              |
|------------------------------|------------------|--------|-----------------------------------------------------------------------------------------|
| handle                 | void*      | in     | ControllerPtr                                                                           |
| return                       | int32_t    | -      | if $>0$, it guarantees devices have processed data. if $<0$, error ocurred.             |


##  AUTDPause (autd3capi)

出力を一時停止する. send functionの一つ. 出力はAUTDResumeで再開できる.

`handle`には`AUTDCreateController`で作成したControllerを使用する.

この関数はエラーが発生した場合に0未満の値を返す.
エラーが生じた場合には`AUTDGetLastError`でエラーメッセージを取得できる.
また,check ackフラグがOn, かつ, 返り値が0より大きい場合は,
データが実際のデバイスで処理されたことを保証する.

| Argument name / return       | type             | in/out | description                                                                              |
|------------------------------|------------------|--------|-----------------------------------------------------------------------------------------|
| handle                 | void*      | in     | ControllerPtr                                                                           |
| return                       | int32_t    | -      | if $>0$, it guarantees devices have processed data. if $<0$, error ocurred.             |


##  AUTDResume (autd3capi)

AUTDPauseで一時停止した出力を再開する.send functionの一つ.

`handle`には`AUTDCreateController`で作成したControllerを使用する.

この関数はエラーが発生した場合に0未満の値を返す.
エラーが生じた場合には`AUTDGetLastError`でエラーメッセージを取得できる.
また,check ackフラグがOn, かつ, 返り値が0より大きい場合は,
データが実際のデバイスで処理されたことを保証する.

| Argument name / return       | type             | in/out | description                                                                              |
|------------------------------|------------------|--------|-----------------------------------------------------------------------------------------|
| handle                 | void*      | in     | ControllerPtr                                                                           |
| return                       | int32_t    | -      | if $>0$, it guarantees devices have processed data. if $<0$, error ocurred.             |

##  AUTDSendGain (autd3capi)

Gainを送信する.send functionの一つ.

`handle`には`AUTDCreateController`で作成したControllerを使用する.

`check_ack`がtrueの場合は,
この関数はデータが実際のデバイスで処理されるまで待機する.

この関数はエラーが発生した場合に0未満の値を返す.
エラーが生じた場合には`AUTDGetLastError`でエラーメッセージを取得できる.
また,check ackフラグがOn, かつ, 返り値が0より大きい場合は,
データが実際のデバイスで処理されたことを保証する.

| Argument name / return       | type             | in/out | description                                                                              |
|------------------------------|------------------|--------|-----------------------------------------------------------------------------------------|
| handle                 | void*      | in     | ControllerPtr                                                                           |
| gain                   | void*      | in     | GainPtr                                                                                 |
| return                       | int32_t    | -      | if $>0$, it guarantees devices have processed data. if $<0$, error ocurred.             |

##  AUTDSendModulation (autd3capi)

Modulationを送信する. send functionの一つ.

`handle`には`AUTDCreateController`で作成したControllerを使用する.

`check_ack`がtrueの場合は,
この関数はデータが実際のデバイスで処理されるまで待機する.

この関数はエラーが発生した場合に0未満の値を返す.
エラーが生じた場合には`AUTDGetLastError`でエラーメッセージを取得できる.
また,check ackフラグがOn, かつ, 返り値が0より大きい場合は,
データが実際のデバイスで処理されたことを保証する.

| Argument name / return       | type             | in/out | description                                                                              |
|------------------------------|------------------|--------|-----------------------------------------------------------------------------------------|
| handle                 | void*      | in     | ControllerPtr                                                                           |
| mod                    | void*      | in     | ModulationPtr                                                                           |
| return                       | int32_t    | -      | if $>0$, it guarantees devices have processed data. if $<0$, error ocurred.             |

##  AUTDSendGainModulation (autd3capi)

GainとModulationを送信する. send functionの一つ.

`handle`には`AUTDCreateController`で作成したControllerを使用する.

`check_ack`がtrueの場合は,
この関数はデータが実際のデバイスで処理されるまで待機する.

この関数はエラーが発生した場合に0未満の値を返す.
エラーが生じた場合には`AUTDGetLastError`でエラーメッセージを取得できる.
また,check ackフラグがOn, かつ, 返り値が0より大きい場合は,
データが実際のデバイスで処理されたことを保証する.

| Argument name / return       | type             | in/out | description                                                                              |
|------------------------------|------------------|--------|-----------------------------------------------------------------------------------------|
| handle                 | void*      | in     | ControllerPtr                                                                           |
| gain                   | void*      | in     | GainPtr                                                                                 |
| mod                    | void*      | in     | ModulationPtr                                                                           |
| return                       | int32_t    | -      | if $>0$, it guarantees devices have processed data. if $<0$, error ocurred.             |

##  AUTDSendSequenceModulation (autd3capi)

PointSequenceとModulationを送信する. send functionの一つ.

`handle`には`AUTDCreateController`で作成したControllerを使用する.

`check_ack`がtrueの場合は,
この関数はデータが実際のデバイスで処理されるまで待機する.

この関数はエラーが発生した場合に0未満の値を返す.
エラーが生じた場合には`AUTDGetLastError`でエラーメッセージを取得できる.
また,check ackフラグがOn, かつ, 返り値が0より大きい場合は,
データが実際のデバイスで処理されたことを保証する.

| Argument name / return       | type             | in/out | description                                                                              |
|------------------------------|------------------|--------|-----------------------------------------------------------------------------------------|
| handle                 | void*      | in     | ControllerPtr                                                                           |
| seq                    | void*      | in     | PointSequencePtr                                                                        |
| mod                    | void*      | in     | ModulationPtr                                                                           |
| return                       | int32_t    | -      | if $>0$, it guarantees devices have processed data. if $<0$, error ocurred.             |

##  AUTDSendGainSequenceModulation (autd3capi)

GainSequenceとModulationを送信する. send functionの一つ.

`handle`には`AUTDCreateController`で作成したControllerを使用する.

`check_ack`がtrueの場合は,
この関数はデータが実際のデバイスで処理されるまで待機する.

この関数はエラーが発生した場合に0未満の値を返す.
エラーが生じた場合には`AUTDGetLastError`でエラーメッセージを取得できる.
また,check ackフラグがOn, かつ, 返り値が0より大きい場合は,
データが実際のデバイスで処理されたことを保証する.

| Argument name / return       | type             | in/out | description                                                                              |
|------------------------------|------------------|--------|-----------------------------------------------------------------------------------------|
| handle                 | void*      | in     | ControllerPtr                                                                           |
| seq                    | void*      | in     | GainSequencePtr                                                                         |
| mod                    | void*      | in     | ModulationPtr                                                                           |
| return                       | int32_t    | -      | if $>0$, it guarantees devices have processed data. if $<0$, error ocurred.             |

##  AUTDSTMController (autd3capi)

STMControllerを取得する.

`handle`には`AUTDCreateController`で作成したControllerを使用する.

この関数を呼び出してから`AUTDFinishSTM`を呼び出すまでの間はhandleの使用は禁止される.

| Argument name / return       | type             | in/out | description                                                                              |
|------------------------------|------------------|--------|-----------------------------------------------------------------------------------------|
| out                    | void**     | out    | pointer to STMControllerPtr                                                             |
| handle                 | void*      | out    | ControllerPtr                                                                        |
| return                       | void       | -      | nothing                                                                                 |

##  AUTDAddSTMGain (autd3capi)

STMControllerにGainを追加する.

`handle`には`AUTDSTMController`で取得したControllerを使用する.

この関数は失敗した場合にfalseを返す.
falseの場合には`AUTDGetLastError`でエラーメッセージを取得できる.

| Argument name / return       | type             | in/out | description                                                                              |
|------------------------------|------------------|--------|-----------------------------------------------------------------------------------------|
| handle                 | void*      | in     | STMControllerPtr                                                                        |
| gain                   | void*      | in     | GainPtr                                                                                 |
| return                       | bool       | -      | true if success                                                                         |

##  AUTDStartSTM (autd3capi)

STMを開始する.

`handle`には`AUTDSTMController`で取得したControllerを使用する.

この関数は失敗した場合にfalseを返す.
falseの場合には`AUTDGetLastError`でエラーメッセージを取得できる.

| Argument name / return       | type             | in/out | description                                                                              |
|------------------------------|------------------|--------|-----------------------------------------------------------------------------------------|
| handle                 | void*      | in     | STMControllerPtr                                                                        |
| freq                   | double     | in     | frequency                                                                               |
| return                       | bool       | -      | true if success                                                                         |

##  AUTDStopSTM (autd3capi)

STMを停止する.

`handle`には`AUTDSTMController`で取得したControllerを使用する.

この関数は失敗した場合にfalseを返す.
falseの場合には`AUTDGetLastError`でエラーメッセージを取得できる.

| Argument name / return       | type             | in/out | description                                                                              |
|------------------------------|------------------|--------|-----------------------------------------------------------------------------------------|
| handle                 | void*      | in     | STMControllerPtr                                                                        |
| return                       | bool       | -      | true if success                                                                         |

##  AUTDFinishSTM (autd3capi)

STMを終了する.

`handle`には`AUTDSTMController`で取得したControllerを使用する.

この関数は失敗した場合にfalseを返す.
falseの場合には`AUTDGetLastError`でエラーメッセージを取得できる.

| Argument name / return       | type             | in/out | description                                                                              |
|------------------------------|------------------|--------|-----------------------------------------------------------------------------------------|
| handle                 | void*      | in     | STMControllerPtr                                                                        |
| return                       | bool       | -      | true if success                                                                         |

##  AUTDEigen3Backend (autd3capi-holo-gain)

Eigen Backendを作成する.

作成したBackendは最終的に`AUTDDeleteBackend`で削除する必要がある.

| Argument name / return       | type             | in/out | description                                                                              |
|------------------------------|------------------|--------|-----------------------------------------------------------------------------------------|
| out                    | void**     | out    | pointer to BackendPtr                                                                   |
| return                       | void       | -      | nothing                                                                                 |

##  AUTDDeleteBackend (autd3capi-holo-gain)

Backendを削除する.

| Argument name / return       | type             | in/out | description                                                                              |
|------------------------------|------------------|--------|-----------------------------------------------------------------------------------------|
| backend                | void*      | in     | BackendPtr                                                                              |
| return                       | void       | -      | nothing                                                                                 |

##  AUTDGainHoloSDP (autd3capi-holo-gain)

holo::SDPを作成する.

`points`は`size`$\times 3$の長さの配列のポインタで,
焦点の位置をx\[0\], y\[0\], z\[0\], x\[1\], \...の順番で指定する.
`amps`は`size`の長さの配列のポインタであり, 焦点の強さを指定する.

作成したGainは最後に`AUTDDeleteGain`で削除する必要がある.

| Argument name / return       | type             | in/out | description                                                                              |
|------------------------------|------------------|--------|-----------------------------------------------------------------------------------------|
| gain                   | void**     | out    | pointer to GainPtr                                                                      |
| backend                | void*      | in     | BackendPtr                                                                              |
| points                 | double*          | in     | pointer to foci array                                                                   |
| amps                   | double*          | in     | pointer to amplitudes array                                                             |
| size                   | int32_t    | in     | number of foci                                                                          |
| alpha                  | double     | in     | parameter                                                                               |
| repeat                 | uint64_t   | in     | parameter                                                                               |
| normalize              | bool       | in     | parameter                                                                               |
| return                       | void       | -      | nothing                                                                                 |

##  AUTDGainHoloEVD (autd3capi-holo-gain)

holo::EVDを作成する.

`points`は`size`$\times 3$の長さの配列のポインタで,
焦点の位置をx\[0\], y\[0\], z\[0\], x\[1\], \...の順番で指定する.
`amps`は`size`の長さの配列のポインタであり, 焦点の強さを指定する.

作成したGainは最後に`AUTDDeleteGain`で削除する必要がある.

| Argument name / return       | type             | in/out | description                                                                              |
|------------------------------|------------------|--------|-----------------------------------------------------------------------------------------|
| gain                   | void**     | out    | pointer to GainPtr                                                                      |
| backend                | void*      | in     | BackendPtr                                                                              |
| points                 | double*          | in     | pointer to foci array                                                                   |
| amps                   | double*          | in     | pointer to amplitudes array                                                             |
| size                   | int32_t    | in     | number of foci                                                                          |
| gamma                  | double     | in     | parameter                                                                               |
| normalize              | bool       | in     | parameter                                                                               |
| return                       | void       | -      | nothing                                                                                 |

##  AUTDGainHoloNaive (autd3capi-holo-gain)

holo::Naiveを作成する.

`points`は`size`$\times 3$の長さの配列のポインタで,
焦点の位置をx\[0\], y\[0\], z\[0\], x\[1\], \...の順番で指定する.
`amps`は`size`の長さの配列のポインタであり, 焦点の強さを指定する.

作成したGainは最後に`AUTDDeleteGain`で削除する必要がある.

| Argument name / return       | type             | in/out | description                                                                              |
|------------------------------|------------------|--------|-----------------------------------------------------------------------------------------|
| gain                   | void**     | out    | pointer to GainPtr                                                                      |
| backend                | void*      | in     | BackendPtr                                                                              |
| points                 | double*          | in     | pointer to foci array                                                                   |
| amps                   | double*          | in     | pointer to amplitudes array                                                             |
| size                   | int32_t    | in     | number of foci                                                                          |
| return                       | void       | -      | nothing                                                                                 |

##  AUTDGainHoloGS (autd3capi-holo-gain)

holo::GSを作成する.

`points`は`size`$\times 3$の長さの配列のポインタで,
焦点の位置をx\[0\], y\[0\], z\[0\], x\[1\], \...の順番で指定する.
`amps`は`size`の長さの配列のポインタであり, 焦点の強さを指定する.

作成したGainは最後に`AUTDDeleteGain`で削除する必要がある.

| Argument name / return       | type             | in/out | description                                                                              |
|------------------------------|------------------|--------|-----------------------------------------------------------------------------------------|
| gain                   | void**     | out    | pointer to GainPtr                                                                      |
| backend                | void*      | in     | BackendPtr                                                                              |
| points                 | double*          | in     | pointer to foci array                                                                   |
| amps                   | double*          | in     | pointer to amplitudes array                                                             |
| size                   | int32_t    | in     | number of foci                                                                          |
| repeat                 | uint64_t   | in     | parameter                                                                               |
| return                       | void       | -      | nothing                                                                                 |

##  AUTDGainHoloGSPAT (autd3capi-holo-gain)

holo::GSPATを作成する.

`points`は`size`$\times 3$の長さの配列のポインタで,
焦点の位置をx\[0\], y\[0\], z\[0\], x\[1\], \...の順番で指定する.
`amps`は`size`の長さの配列のポインタであり, 焦点の強さを指定する.

作成したGainは最後に`AUTDDeleteGain`で削除する必要がある.

| Argument name / return       | type             | in/out | description                                                                              |
|------------------------------|------------------|--------|-----------------------------------------------------------------------------------------|
| gain                   | void**     | out    | pointer to GainPtr                                                                      |
| backend                | void*      | in     | BackendPtr                                                                              |
| points                 | double*          | in     | pointer to foci array                                                                   |
| amps                   | double*          | in     | pointer to amplitudes array                                                             |
| size                   | int32_t    | in     | number of foci                                                                          |
| repeat                 | uint64_t   | in     | parameter                                                                               |
| return                       | void       | -      | nothing                                                                                 |

##  AUTDGainHoloLM (autd3capi-holo-gain)

holo::LMを作成する.

`points`は`size`$\times 3$の長さの配列のポインタで,
焦点の位置をx\[0\], y\[0\], z\[0\], x\[1\], \...の順番で指定する.
`amps`は`size`の長さの配列のポインタであり, 焦点の強さを指定する.

作成したGainは最後に`AUTDDeleteGain`で削除する必要がある.

| Argument name / return       | type             | in/out | description                                                                              |
|------------------------------|------------------|--------|-----------------------------------------------------------------------------------------|
| gain                   | void**     | out    | pointer to GainPtr                                                                      |
| backend                | void*      | in     | BackendPtr                                                                              |
| points                 | double*          | in     | pointer to foci array                                                                   |
| amps                   | double*          | in     | pointer to amplitudes array                                                             |
| size                   | int32_t    | in     | number of foci                                                                          |
| eps_1                  | double     | in     | parameter                                                                               |
| eps_2                  | double     | in     | parameter                                                                               |
| tau                    | double     | in     | parameter                                                                               |
| k_max                  | uint64_t   | in     | parameter                                                                               |
| initial                | double*    | in     | parameter                                                                               |
| initial_size           | int32_t    | in     | length of initial                                                                       |
| return                       | void       | -      | nothing                                                                                 |

##  AUTDGainHoloGaussNewton (autd3capi-holo-gain)

holo::GaussNewtonを作成する.

`points`は`size`$\times 3$の長さの配列のポインタで,
焦点の位置をx\[0\], y\[0\], z\[0\], x\[1\], \...の順番で指定する.
`amps`は`size`の長さの配列のポインタであり, 焦点の強さを指定する.

作成したGainは最後に`AUTDDeleteGain`で削除する必要がある.

| Argument name / return       | type             | in/out | description                                                                              |
|------------------------------|------------------|--------|-----------------------------------------------------------------------------------------|
| gain                   | void**     | out    | pointer to GainPtr                                                                      |
| backend                | void*      | in     | BackendPtr                                                                              |
| points                 | double*          | in     | pointer to foci array                                                                   |
| amps                   | double*          | in     | pointer to amplitudes array                                                             |
| size                   | int32_t    | in     | number of foci                                                                          |
| eps_1                  | double     | in     | parameter                                                                               |
| eps_2                  | double     | in     | parameter                                                                               |
| k_max                  | uint64_t   | in     | parameter                                                                               |
| initial                | double*    | in     | parameter                                                                               |
| initial_size           | int32_t    | in     | length of initial                                                                       |
| return                       | void       | -      | nothing                                                                                 |

##  AUTDGainHoloGradientDescent (autd3capi-holo-gain)

holo::GradientDescentを作成する.

`points`は`size`$\times 3$の長さの配列のポインタで,
焦点の位置をx\[0\], y\[0\], z\[0\], x\[1\], \...の順番で指定する.
`amps`は`size`の長さの配列のポインタであり, 焦点の強さを指定する.

作成したGainは最後に`AUTDDeleteGain`で削除する必要がある.

| Argument name / return       | type             | in/out | description                                                                              |
|------------------------------|------------------|--------|-----------------------------------------------------------------------------------------|
| gain                   | void**     | out    | pointer to GainPtr                                                                      |
| backend                | void*      | in     | BackendPtr                                                                              |
| points                 | double*          | in     | pointer to foci array                                                                   |
| amps                   | double*          | in     | pointer to amplitudes array                                                             |
| size                   | int32_t    | in     | number of foci                                                                          |
| eps                    | double     | in     | parameter                                                                               |
| step                   | double     | in     | parameter                                                                               |
| k_max                  | uint64_t   | in     | parameter                                                                               |
| initial                | double*    | in     | parameter                                                                               |
| initial_size           | int32_t    | in     | length of initial                                                                       |
| return                       | void       | -      | nothing                                                                                 |

##  AUTDGainHoloAPO (autd3capi-holo-gain)

holo::APOを作成する.

`points`は`size`$\times 3$の長さの配列のポインタで,
焦点の位置をx\[0\], y\[0\], z\[0\], x\[1\], \...の順番で指定する.
`amps`は`size`の長さの配列のポインタであり, 焦点の強さを指定する.

作成したGainは最後に`AUTDDeleteGain`で削除する必要がある.

| Argument name / return       | type             | in/out | description                                                                              |
|------------------------------|------------------|--------|-----------------------------------------------------------------------------------------|
| gain                   | void**     | out    | pointer to GainPtr                                                                      |
| backend                | void*      | in     | BackendPtr                                                                              |
| points                 | double*          | in     | pointer to foci array                                                                   |
| amps                   | double*          | in     | pointer to amplitudes array                                                             |
| size                   | int32_t    | in     | number of foci                                                                          |
| eps                    | double     | in     | parameter                                                                               |
| lambda                 | double     | in     | parameter                                                                               |
| k_max                  | uint64_t   | in     | parameter                                                                               |
| return                       | void       | -      | nothing                                                                                 |

##  AUTDGainHoloGreedy (autd3capi-holo-gain)

holo::Greedyを作成する.

`points`は`size`$\times 3$の長さの配列のポインタで,
焦点の位置をx\[0\], y\[0\], z\[0\], x\[1\], \...の順番で指定する.
`amps`は`size`の長さの配列のポインタであり, 焦点の強さを指定する.

作成したGainは最後に`AUTDDeleteGain`で削除する必要がある.

| Argument name / return       | type             | in/out | description                                                                              |
|------------------------------|------------------|--------|-----------------------------------------------------------------------------------------|
| gain                   | void**     | out    | pointer to GainPtr                                                                      |
| backend                | void*      | in     | BackendPtr                                                                              |
| points                 | double*          | in     | pointer to foci array                                                                   |
| amps                   | double*          | in     | pointer to amplitudes array                                                             |
| size                   | int32_t    | in     | number of foci                                                                          |
| phase_div              | int32_t    | in     | parameter                                                                               |
| return                       | void       | -      | nothing                                                                                 |

##  AUTDModulationRawPCM (autd3capi-from-file-modulation)

modulation::RawPCMを作成する.

作成したModulationは最後に`AUTDDeleteModulation`で削除する必要がある.

| Argument name / return       | type             | in/out | description                                                                              |
|------------------------------|------------------|--------|-----------------------------------------------------------------------------------------|
| mod                    | void**     | out    | pointer to ModulationPtr                                                                |
| filename               | char*            | in     | pointer to filename string                                                              |
| sampling_freq          | double     | in     | sampling frequency of PCM                                                               |
| mod_sampling_freq_div  | uint16_t   | in     | Sampling frequency division                                                             |
| return                       | void       | -      | nothing                                                                                 |

##  AUTDModulationWav (autd3capi-from-file-modulation)

modulation::Wavを作成する.

作成したModulationは最後に`AUTDDeleteModulation`で削除する必要がある.

| Argument name / return       | type             | in/out | description                                                                              |
|------------------------------|------------------|--------|-----------------------------------------------------------------------------------------|
| mod                    | void**     | out    | pointer to ModulationPtr                                                                |
| filename               | char*            | in     | pointer to filename string                                                              |
| mod_sampling_freq_div  | uint16_t   | in     | Sampling frequency division                                                             |
| return                       | void       | -      | nothing                                                                                 |

##  AUTDLinkTwinCAT (autd3capi-twincat-link)

link::TwinCATを作成する.

| Argument name / return       | type             | in/out | description                                                                              |
|------------------------------|------------------|--------|-----------------------------------------------------------------------------------------|
| out                    | void**     | out    | pointer to LinkPtr                                                                      |
| return                       | void       | -      | nothing                                                                                 |

##  AUTDLinkRemoteTwinCAT (autd3capi-remote-twincat-link)

link::RemoteTwinCATを作成する.

| Argument name / return       | type             | in/out | description                                                                              |
|------------------------------|------------------|--------|-----------------------------------------------------------------------------------------|
| out                    | void**     | out    | pointer to LinkPtr                                                                      |
| remote_ip_addr         | char*            | in     | pointer to remote ip address                                                            |
| remote_ams_net_id      | char*            | in     | pointer to remote ams net id                                                            |
| local_ams_net_id       | char*            | in     | pointer to local ams net id                                                             |
| return                       | void       | -      | nothing                                                                                 |

##  AUTDGetAdapterPointer (autd3capi-soem-link)

EtherCAT adapter information listへのポインタを取得する.

EtherCAT adapter informationの取得は`AUTDGetAdapter`で行う.

取得したポインタは最後に`AUTDFreeAdapterPointer`で開放する必要がある.

この関数は取得したlistのサイズを返す.

| Argument name / return       | type             | in/out | description                                                                              |
|------------------------------|------------------|--------|-----------------------------------------------------------------------------------------|
| out                    | void**     | out    | pointer to pointer to adapter information list                                          |
| return                       | int32_t    | -      | length of EtherCAT adapter information list                                             |

##  AUTDGetAdapter (autd3capi-soem-link)

EtherCAT adapter informationを取得する.

`p_adapter`は`AUTDGetAdapterPointer`で取得したポインタを渡す.

`desc`と`name`には長さ128のバッファを渡せば十分である.

| Argument name / return       | type             | in/out | description                                                                              |
|------------------------------|------------------|--------|-----------------------------------------------------------------------------------------|
| p_adapter              | void*      | in     | pointer to EtherCAT adapter information list                                            |
| index                  | int32_t    | in     | adapter index                                                                           |
| desc                   | char*      | in     | adapter description                                                                     |
| name                   | char*      | in     | adapter name                                                                            |
| return                       | void       | -      | nothing                                                                                 |

##  AUTDFreeAdapterPointer (autd3capi-soem-link)

`AUTDGetAdapterPointer`で取得したポインタを開放する.

| Argument name / return       | type             | in/out | description                                                                              |
|------------------------------|------------------|--------|-----------------------------------------------------------------------------------------|
| p_adapter              | void*      | in     | pointer to EtherCAT adapter information list                                            |
| return                       | void       | -      | nothing                                                                                 |

##  AUTDLinkSOEM (autd3capi-soem-link)

link::SOEMを作成する.

`ifname`にはインターフェース名を入れる. これは, `AUTDGetAdapter`で取得できる.

| Argument name / return       | type             | in/out | description                                                                              |
|------------------------------|------------------|--------|-----------------------------------------------------------------------------------------|
| out                    | void**     | out    | pointer to LinkPtr                                                                      |
| ifname                 | int32_t    | in     | number of devices                                                                       |
| cycle_ticks            | uint32_t   | in     | cycle ticks                                                                             |
| return                       | void       | -      | nothing                                                                                 |

##  AUTDSetSOEMOnLost (autd3capi-soem-link)

SOEM linkにOnLostCallbackを設定する.

linkには`AUTDLinkSOEM`で作成したものを使う.

ErrorHandlerは`void (*)(const char*)`と定義されており,
エラーが発生したときにエラーメッセージを引数に呼ばれる.

| Argument name / return       | type             | in/out | description                                                                              |
|------------------------------|------------------|--------|-----------------------------------------------------------------------------------------|
| link                   | void*      | in     | SOEM LinkPtr                                                                            |
| handler                | ErrorHandler     | in     | error handler                                                                           |
| return                       | void       | -      | nothing                                                                                 |

##  AUTDLinkEmulator (autd3capi-emulator-link)

link::Emulatorを作成する.

`cnt`には`AUTDCreateController`で作成したControllerを渡す.

| Argument name / return       | type             | in/out | description                                                                              |
|------------------------------|------------------|--------|-----------------------------------------------------------------------------------------|
| out                    | void**     | out    | pointer to LinkPtr                                                                      |
| port                   | uint16_t   | in     | number of devices                                                                       |
| cnt                    | void*      | in     | ControllerPtr                                                                           |
| return                       | void       | -      | nothing                                                                                 |
