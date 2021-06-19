# 1.4.0
* Update firmware version to 1.2
    * Change the meaning of duty ratio and phase
        * Duty ratio is now calculated by (`duty`+1)/512 %
        * Phase is now calculated by `phase`/256 rad
* Add Controller::puase() and Controller::resume()
* Add Controller::set_enable()

# 1.3.0
* Add functions to specify (relative) amplitude in Modulations
* Update firmware version to 1.1
    * Maximum modulation buffer size is now 65536
    * Maximum sequence point size is now 65536
    * Maximum delay is now 255
* Remove Controller::synchronize()
    * This function is no more required

# 1.2.0
* Fix #3
* Improve HoloGreedy performance
* Fix modulation::Sine  
* Add modulation::SinePressure
* Delete modulation::Saw
* Add attenuation setting in capi 

# 1.1.0
* Make LinkPtr unique
    * With this change, implementation of timers and the APIs around STM were changed.
* Make GeometryPtr unique
* Add force fan flag
* Add HoloGainGreedy

# 1.0.0
* Major revisions have been performed, APIs have also been changed massively.

# 0.9.3
* Add `Rebuild()` method to Gain
* Change the declaration of HoloGain. Selection by OPT_METHOD is deleted and introduced HoloGainSDP, HoloGainEVD, HoloGainNaive, HoloGainGS, HoloGainGSPAT, HoloGainLM classes.

# 0.9.2
* Make double-precision float default
    * `USE_DOUBLE_AUTD` was removed, and `USE_SINGLE_FLOAT_AUTD` was added
    * capi still use single-precision float in its interface, but it uses double internally
* Fix issue #2

# 0.9.1
* Fix bugs in `autdsoem`
* Make Eigen enable by default
* Add `HoloGainE` and `HoloGainB` alias for `HoloGain<Eigen3Backend>` and `HoloGain<BLASBackend>`, respectively
* Add `Geometry::DelDevice` and `Geometry::ClearDevices`
* Functions that may fail now return a `Result` value
* `Calibrate` is now deleted, please use `Synchronize` instead
* Add a function to get the error that occurred last in capi
* Rename `AppendSTMGain` to `AddSTMGain`
* Rename `AppendPoint` to `AddPoint`
* Fix issue #1

# 0.9.0
* Add BLAS backend in HoloGain
* Delete `SetDelay` because of technical issue in FPGA
* Add `vector.hpp`, `matrix.hpp`, and `quaternion.hpp` to build without Eigen3
* Subdivide the main project into some small projects
* Add some CMake options to finely control build options
    * Now, some projects are disabled by default
* Fix bug which requires installing npcap (wpcap.dll) even when SOEM is not used (Windows) 

# 0.8.1
* Provide win-x86, linux-arm32, linux-arm64 binaries

# 0.8.0
* Delete backward compatibility
* Delete `Geometry::DelDevice()` and `Geometry::deviceIdForDeviceIdx()`
* Delete `Controller::LateralModulationAT()`
* Delete `vector3.hpp` and `quaternion.hpp`
* Rename some functions to unify the naming
* Add `DebugLink`
* Add `SetDelay`
* Add `Geometry::wavelength()` and `Geometry::set_wavelength()`
* Add `USE_DOUBLE_AUTD` macro to switch float type 
    * Set default float type to `float`
    * Also, change float type in capi from `double` to `float`
* Perform overall refactoring

# 0.7.2
* Support Apple Silicon mac

# 0.7.1
* Changed Timer implementation from TimerQueue to timeSetEvent (Windows)
    * Because TimerQueue  does not work with the resolution of 16ms or less
* Improved speed of HoloGain with OptMethod::LM

# 0.7
* Add `SquareModulation`
* Add `CustomModulation` in capi
* Extended modulation buffer size to maximum 32000
* Sampling frequency become mutable
* Implemented some new methods of generating multi foci
* Some Gain amplitude arguments become duty ratio. If you want to specify the output in sound pressure as before, use an overload of double type.

# 0.6.2
* Implemented backward compatibility

# 0.6.1
* Fixed a minor bug

# 0.6.0
* Fixed a bug of phase discontinuity in firmware
* Improved performance of Point Sequence Mode

# 0.5.0
* Fixed some bugs
* Added Sequence in CAPI

# 0.5.0-rc2
* Fixed bug in AM

# 0.5.0-rc1
* Deleted deprecated items
    * Controller::Open(), LinkType
* Renamed CalibrateModulation to Calibrate
* Moved gains to autd::gain namespace
* Moved modulations to autd::modulation namespace
* Moved soem_link to external static library
* Moved ethercat_link to external static library
    * And rename EtherCATLink to TwinCATLink
* Unified internal pointer type to a smart pointer
    * In c-api, these are wrapped by struct.
* Implemented Point Sequence Mode

# 0.4.1
* Changed the way to open link.
    - Made Controller::Open deprecated.
* Added EmulatorLink (experimental).
* Fixed bug in reading with TwinCAT
    - Updated AUTDServer
* Added sample code

# 0.4.0
* Added firmware_info_list()
* Improved CalibrateModulation method

# 0.3.2
* Added WavModulation
* Delete unused methods

# 0.3.1
* Delete Eigen3 from interface
* Delete deprecated methods

# 0.3.0
* introduced versioning
    *The meanings of version number 3.x.y.z are
        * x: Architecture version.
        * y: Firmware version.
        * z: Software version.
* Extensive refactoring
    * Made code conform to almost Google C ++ Style Guide
       * There are some differences from original. Refer to client/readme.md for details
    * Removed pimpl idiom
* (autd3sharp) Made C# library compliant with .Net Standard 2.0
* (autd3sharp) register C# wrapper to NuGet https://www.nuget.org/packages/autd3sharp
    * Therefore, autd3sharp is removed from this repository
* (pyautd) supports installing with pip
* (ruautd) fix implementations

# 0.2.4 (software/cpu)
* fix bug in closing
* fix bug of data packet loss when using SOEM

# 0.2.3 (software)
* change implementation of LateralModulation
* rename LateralModulation to Spatio-Temporal Modulation
* fix bugs in timer.cpp (Linux/Max)

# 0.2.2 (software/cpu/fpga)
* fix bug in closing
* fpga mcs file compression & configuration time reduction

# 0.2.1 (software)
* fix bug when using SOEM
    * make calling ec_send_processdata() exclusive
* add Rust version (future/autd)

# 0.2.0 (software/cpu/fpga)
* implement ModulationCalibration()
    * This caused breaking changes in software/CPU/FPGA
    * default Sync0 period is set to 1ms

# 0.1.3
* fix AppendGain()
* Sync0 period is set to 64ms

# 0.1.2
* fix modulation synchronization probrem (100ns)

# 0.1.1
* add Sync0 synchronization signal with SOEM
* fix modulation synchronization problem (100us)

# 0.1.0
* add a feature that controlling by Simple Open EtherCAT Master (SOEM)
    * Linux and Mac support
    * It is no longer necessary to use TwinCAT on Windows
    * USB to Ethernet conversion cable is available
* delete dependency on boost library