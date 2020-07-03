# 0.5.0
* Fix some bugs
* Add Sequence in CAPI

# 0.5.0-rc2
* Fix bug in AM

# 0.5.0-rc1
* Delete deprecated items
    * Controller::Open(), LinkType
* Rename CalibrateModulation to Calibrate
* Move gains to autd::gain namespace
* Move modulations to autd::modulation namespace
* Move soem_link to external static library
* Move ethercat_link to external static library
    * And rename EtherCATLink to TwinCATLink
* Unify internal pointer type to a smart pointer
    * In c-api, these are wrapped by struct.
* Implement Point Sequence Mode

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