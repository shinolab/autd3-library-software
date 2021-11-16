# Link

_Link_ is the interface to the AUTD3 device.
You need to choose one of the following.

## TwinCAT

TwinCAT is the only official way to use EherCAT on a PC.
TwinCAT is a very special software which only supports Windows and makes Windows real-time.

TwinCAT requires a specific network controller, please check [supported network controllers list](https://infosys.beckhoff.com/english.php?content=../content/1033/tc3_overview/9309844363.html&id=).

> Note: Alternatively, after installing TwinCAT, you can find the _Vendor ID_ and _Device ID_ of the supported network controller in `C:/TwinCAT/3.1/Driver/System/TcI8254x.inf`.

### How to install TwinCAT

TwinCAT cannot coexist with Hyper-V and Virtual Machine Platform.
Therefore, you need to disable these features.
To do this, for example, run PowerShell with administrative privileges, and enter following commands,
```
Disable-WindowsOptionalFeature -Online -FeatureName Microsoft-Hyper-V-Hypervisor
Disable-WindowsOptionalFeature -online -featurename VirtualMachinePlatform
```

First, download _TwinCAT XAE_ from the [official website](https://www.beckhoff.com/en-en/).
Registration (free) is required to download the software.

Then, launch the installer and follow the instructions.
**At this time, check "TwinCAT XAE Shell install" and uncheck "Visual Studio Integration".**

After the installation, reboot and then run `C:/TwinCAT/3.1/System/win8settick.bat` with administrator privileges and reboot again.

Finally, copy the file `AUTDServer/AUTD.xml` in the SDK to `C:/TwinCAT/3.1/Config/Io/EtherCAT`.

### AUTDServer

To use TwinCAT Link, run `AUTDServer/AUTDServer.exe` in advance.
After executing AUTDServer, you will be prompted to enter your IP address, but please leave it empty.
Then, a TwinCAT XAE Shell will be launched.
Finally, you will be asked to close the shell.
If you use AUTDServer at the first time, please enter `No` and follow the instructions below to setup 

> Note: If you close it, you can open AUTDServer shell by launching `%Temp%/TwinCATAUTDServer/TwinCATAUTDServer.sln` as TcXaeShell Application, where `%Temp%` is an environment variable, usually `C:/Users/(user name)/AppData/Local/Temp`.

Note that AUTDServer (i.e. TwinCAT) will break the link when you turn off your PC, enter the sleep mode, etc., so you should re-run AUTDServer each time.

#### Install Driver

For the first time, you need to install a driver for EtherCAT.
From the top menu of the TwinCAT XAE Shell, go to "TwinCAT" -> "Show Realtime Ethernet Compatible Devices" and select a compatible device from the list of compatible devices and click on Install.
If nothing is shown in the compatible devices, the network controllers of the PC are not compatible with TwinCAT.

#### Licensing

The first time you run the program, you will get an error related to the license, so open "Solution Explorer" -> "SYSTEM" -> "License" in XAE Shell, and click "7 Days Trial License ...". and enter the text displayed on the screen.
Note that the license is a 7-day trial license, but can be reissued when it expires by performing the same procedure again.
After the license is issued, close the XAE Shell and run "AUTDServer.exe" again.

### Trouble shooting

If you try to use a large number of devices, you may get an error like the figure below.
In this case, multiply the values of `TaskCycleTime` and `Sync0Cycletime` in `settings.ini` by an integer, and run AUTDServer again.
Roughly speaking, `TaskCycleTime` means the interval of data update, and `Sync0Cycletime` means the interval of synchronization signal firing.
Therefore, you should choose as small values as possible.
How many times the value depends on the number of connected devices.
For example, when you have 9 devices, it should work if you multiply the number by 2.

```
TaskCycleTime=20000
CPUbaseTime=10000
Sync0Cycletime=1000000
```

<figure>
  <img src="https://raw.githubusercontent.com/shinolab/autd3-library-software/master/book/src/fig/Users_Manual/tcerror.jpg"/>
  <figcaption>TwinCAT error when using 9 devices</figcaption>
</figure>

## RemoteTwinCAT

As mentioned above, TwinCAT requires a Windows OS and a specific network adapter.
If you want to use develop client applications on non-Windows PCs, you can use RemoteTwinCAT link to control TwinCAT remotely.
(SOEM link described below also works on cross-platform).

To use RemoteTwinCAT, you need to prepare two PCs.
One of the PCs must be able to use TwinCAT link described above.
Herein, we call this PC a "server".
On the other hand, the PC on the development side, i.e., the one that uses the SDK, has no restriction, but is required to be connected to the same LAN as the server, which we call the "client".

First, please connect the "server" to the AUTD device.
In this case, the LAN adapter used must be TwinCAT compatible, as in the TwinCAT link.
Also, connect the server and the client with another LAN.
This LAN adapter does not need to be TwinCAT compatible [^fn_remote_twin].
Then, check the IP of the LAN between the server and the client.
For example, let's assume that the server IP is "169.254.205.219" and the client IP is "169.254.175.45".
Next, start AUTDServer on the "server".
After starting, you will be asked to enter the IP, where you should enter the IP of the client side (`169.254.175.45` in this example).
Then, enter "No" at the end to leave TwinCATAUTDServer open.
As shown in the following figure, open "SYSTEM"→"Routes" and check _AmsNetId_ in Current Route tab and _Local NetId_ in NetId Management tab.

<figure>
  <img src="https://raw.githubusercontent.com/shinolab/autd3-library-software/master/book/src/fig/Users_Manual/Current_Route.jpg"/>
  <img src="https://raw.githubusercontent.com/shinolab/autd3-library-software/master/book/src/fig/Users_Manual/NetId_Management.jpg"/>
  <figcaption>AmsNetId/Local NetId</figcaption>
</figure>

Let's assume that _AmsNetId_ and _Local NetId_ are "169.254.175.45.1.1" and "172.16.99.194.1.1", respectively.
In this case, the client side should include the `autd3/link/remote_twincat.hpp` header, and use RemoteTwinCAT link as,
```cpp
#include "autd3/link/remote_twincat.hpp"

...
  const std::string remote_ipv4_addr = "169.254.205.219";
  const std::string remote_ams_net_id = "172.16.99.194.1.1";
  const std::string local_ams_net_id = "169.254.175.45.1.1";
  auto link = autd::link::RemoteTwinCAT::create(remote_ipv4_addr, remote_ams_net_id, local_ams_net_id);
```

If you get TCP-related errors, there is a possibility that the ADS protocol is blocked by firewall.
If so, configure your firewall to allow connections on port 48898 of TCP/UDP.

## SOEM

[SOEM](https://github.com/OpenEtherCATsociety/SOEM) is an open-source EherCAT Master library.
Unlike TwinCAT, SOEM runs on ordinary Windows, so its real-time performance is not guaranteed.
Therefore, it is recommended to use TwinCAT in principle.
SOEM should be used only when there is a compelling reason to use it, or only during development.
On the other hand, SOEM has the advantage of cross-platform operation and simple installation.

For Windows, you should install [npcap](https://nmap.org/npcap/) or [WinPcap](https://www.winpcap.org/) in advance.
npcap is the successor of WinPcap, and we recommend you to use it.
**Note that npcap should be installed with "WinPcap API compatible mode".**
For Linux/macOS, no special preparation is required.

Please include `autd3/link/soem.hpp` to use SOEM link.
```cpp
#include "autd3/link/soem.hpp"

...

  auto link = autd::link::SOEM::create(ifname, autd->geometry()->num_devices());
```
The first argument of `SOEM::create` is the interface name and the second argument is the number of devices.
The interface name is the name of the ethernet interface connected to the AUTD3 device.
The interface list can be retrieved by the `SOEM::enumerate_adapters` function.
```cpp
  const auto adapters = autd::link::SOEM::enumerate_adapters();
  for (auto&& [desc, name] : adapters) std::cout << desc << ", " << name << std::endl;
```

Note that SOEM sometimes becomes unstable when a large number of devices are used.
In this case, increase the value of the third argument of `create` (`cycle_ticks`) (default is 1).
```cpp
  const uint32_t cycle_ticks = 2;
  auto link = autd::link::SOEM::create(ifname, autd->geometry()->num_devices(), cycle_ticks);
```
The `cycle_ticks` controls the data update interval and the firing period of the synchronization signal.
The default values of the data update interval and the firing period of the synchronization signal are $\SI{1000}{μs}$ and $\SI{500}{μs}$, respectively, which are multiplied by the value of `cycle_ticks`.
Therefore, it is recommended to choose the smallest possible value.

SOEM Link can also be configured with a callback when an unrecoverable error occurs (e.g., a cable is disconnected)[^fn_soem_err].
The callback takes an error message as an argument.
```cpp
  link->on_lost([](const std::string& msg) {
    std::cerr << "Link is lost\n";
    std::cerr << msg;
    std::quick_exit(-1);
  });
```

## Emulator

Emulator link is a link to use [AUTD Emulator](https://github.com/shinolab/autd-emulator) (which will be described after).

Before using it, you need to run the AUTD Emulator.

To use Emulator link, include the `autd3/link/emulator.hpp` .
```cpp
#include "autd3/link/emulator.hpp"

...

  auto link = autd::link::Emulator::create(50632, autd->geometry());
```
The first argument of `Emulator::create` is a port number (default is 50632), and the second argument is Geometry.

[^fn_remote_twin]: Wireless LAN also can be used.

[^fn_soem_err]: Since it is not recoverable, the only thing you can do is terminate the program immediately.

