/*
 * File: SimpleExample_SOEM.cs
 * Project: Test
 * Created Date: 25/08/2019
 * Author: Shun Suzuki
 * -----
 * Last Modified: 20/02/2020
 * Modified By: Shun Suzuki (suzuki@hapis.k.u-tokyo.ac.jp)
 * -----
 * Copyright (c) 2019 Hapis Lab. All rights reserved.
 * 
 */

using AUTD3Sharp;
using System;
using System.Linq;
using System.Threading;

namespace AUTD3SharpTest.Test
{
    internal class SimpleExample_SOEM
    {
        public static void Test()
        {
            Console.WriteLine("Start Simple Test (SOEM)");

            double x = AUTD.AUTDWidth / 2;
            double y = AUTD.AUTDHeight / 2;
            double z = 150;

            using (AUTD autd = new AUTD())
            {
                // AddDevice() must be called before Open(), and be called as many times as the number of AUTDs connected.
                autd.AddDevice(Vector3d.Zero, Vector3d.Zero);
                //autd.AddDevice(Vector3d.UnitY * AUTD.AUTDHeight, Vector3d.Zero);

                System.Collections.Generic.IEnumerable<EtherCATAdapter> adapters = AUTD.EnumerateAdapters();
                foreach ((EtherCATAdapter adapter, int index) in adapters.Select((adapter, index) => (adapter, index)))
                {
                    Console.WriteLine($"[{index}]: {adapter}");
                }

                Console.Write("Choose number: ");
                int i;
                while (!int.TryParse(Console.ReadLine(), out i)) { }
                string ifname = adapters.ElementAt(i).Name;

                autd.Open(LinkType.SOEM, ifname);
                // If you have already recognized the EtherCAT adapter name, you can write it directly like below.
                // autd.Open(LinkType.SOEM, "\\Device\\NPF_{D8BC5907-A0E5-4EAF-A013-8C7F76E3E1F3}");

                // If you use more than one AUTD, call this function only once after Open().
                // It takes several seconds proportional to the number of AUTD you use.
                //autd.CalibrateModulation();

                // AM
                Console.WriteLine("Amplitude Modulation");

                //Gain gain = AUTD.FocalPointGain(x, y, z); // Focal point @ (x, y, z) [mm]
                //autd.AppendGainSync(gain);

                Vector3d[] focuses = new[] {
                    new Vector3d(x - 30, y ,z),
                    new Vector3d(x + 30, y ,z)
                };
                double[] amps = new[] {
                    1.0,
                    1.0
                };
                autd.AppendGainSync(AUTD.HoloGain(focuses, amps));

                Modulation mod = AUTD.SineModulation(150); // AM sin 150 Hz
                autd.AppendModulationSync(mod);

                Console.WriteLine("press any key to start spatio-temporal modulation...");
                Console.ReadKey(true);

                // STM
                Console.WriteLine("Spatio-Temporal Modulation");
                autd.AppendModulationSync(AUTD.Modulation(255));

                Gain f1 = AUTD.FocalPointGain(x + 2.5, y, z);
                Gain f2 = AUTD.FocalPointGain(x - 2.5, y, z);

                autd.AppendSTMGain(f1);
                autd.AppendSTMGain(f2);
                autd.StartSTModulation(100); // Tapping f1 and f2 at 100Hz

                Console.WriteLine("press any key to finish...");
                Console.ReadKey(true);

                autd.FinishSTModulation();
            }
        }
    }
}
