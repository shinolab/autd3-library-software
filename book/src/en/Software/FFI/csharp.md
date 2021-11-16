# C\#

[autd3sharp](https://github.com/shinolab/autd3sharp) provides a wrapper for **.Net Standard 2.1**.

## Installation

It is available at [NuGet](https://www.nuget.org/packages/autd3sharp), and you can install it from NuGet.

### Installation for Unity

If you want to use it from Unity, please use _unitypackage_ which is available at [GitHub Release](https://github.com/shinolab/autd3sharp/releases).

After installing this package, go to `Project Settings > Player` and check `Allow 'unsafe' code`. 
Also, to suppress warnings, add `-nullable:enable` to `Additional Compiler Arguments`.

**Note that the Unity version has a left-handed coordinate system with z-axis reversed, and the unit of distance is $\SI{}{m}$.**

## Usage

Basically, it is designed to be the same as the C++ version.

For example, the equivalent code of [Getting Started](./Users_Manual/getting_started.md) is the following.

```csharp
using AUTD3Sharp;
using AUTD3Sharp.Utils;

namespace example
{
    internal class Program
    {
        public static string GetIfname()
        {
            var adapters = AUTD.EnumerateAdapters();
            var etherCATAdapters = adapters as EtherCATAdapter[] ?? adapters.ToArray();
            foreach (var (adapter, index) in etherCATAdapters.Select((adapter, index) => (adapter, index)))
                Console.WriteLine(\$"[{index}]: {adapter}");

            Console.Write("Choose number: ");
            int i;
            while (!int.TryParse(Console.ReadLine(), out i)) { }
            return etherCATAdapters.ElementAt(i).Name;
        }

        public static void Main()
        {
            var autd = new AUTD();
            autd.AddDevice(Vector3d.Zero, Vector3d.Zero);

            var ifname = GetIfname();
            var link = Link.SOEM(ifname, autd.NumDevices);
            if (!autd.Open(link))
            {
                Console.WriteLine(AUTD.LastError);
                return;
            }

            autd.Clear();

            var firmList = autd.FirmwareInfoList().ToArray();
            foreach (var (firm, index) in firmList.Select((firm, i) => (firm, i)))
                Console.WriteLine(\$"AUTD {index}: {firm}");

            autd.SilentMode = false;

            const double x = AUTD.TransSpacing * ((AUTD.NumTransInX - 1) / 2.0);
            const double y = AUTD.TransSpacing * ((AUTD.NumTransInY - 1) / 2.0);
            const double z = 150.0;
            var g = Gain.FocalPoint(new Vector3d(x, y, z));
            var m = Modulation.Sine(150);
            autd.Send(g, m);

            Console.ReadKey(true);

            autd.Close();
        }
    }
}
```

For a more detailed example, see [autd3sharp's example](https://github.com/shinolab/autd3sharp/tree/master/example).

## Trouble shooting

Q. Cannot run from linux or macOS

A. Run as root

```
sudo dotnet run
```

---

Q. Cannot run from Ubuntu 20.04

A. Specify runtime

```
sudo dotnet run -r ubuntu-x64
```

---

Q. Cannot be used from .Net framework

A. .Net framework is not supported. If you copy and paste the whole source code, it may work.

---

If you have any other questions, please send them to [GitHub issues](https://github.com/shinolab/autd3sharp/issues).
