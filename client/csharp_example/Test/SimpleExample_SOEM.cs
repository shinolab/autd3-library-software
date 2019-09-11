/*
 * File: SimpleExample_SOEM.cs
 * Project: Test
 * Created Date: 25/08/2019
 * Author: Shun Suzuki
 * -----
 * Last Modified: 11/09/2019
 * Modified By: Shun Suzuki (suzuki@hapis.k.u-tokyo.ac.jp)
 * -----
 * Copyright (c) 2019 Hapis Lab. All rights reserved.
 * 
 */

using AUTD3Sharp;
using System;
using System.Linq;

namespace AUTD3SharpTest.Test
{
    internal class SimpleExample_SOEM
    {
        public static void Test()
        {
            Console.WriteLine("Start Simple Test (SOEM)");

            float x = AUTD.AUTDWidth / 2;
            float y = AUTD.AUTDHeight / 2;
            float z = 150;

            using (AUTD autd = new AUTD())
            {
                // AddDevice() must be called before Open(), and be called as many times as the number of AUTDs connected.
                autd.AddDevice(Vector3f.Zero, Vector3f.Zero);
                //autd.AddDevice(Vector3f.UnitY * AUTD.AUTDHeight, Vector3f.Zero);

                var adapters = AUTD.EnumerateAdapters();
                foreach (var (adapter, index) in adapters.Select((adapter, index) => (adapter, index)))
                {
                    Console.WriteLine($"[{index}]: {adapter}");
                }

                Console.Write("Choose number: ");
                int i;
                while (!int.TryParse(Console.ReadLine(), out i)) { }
                string ifname = adapters.ElementAt(i).Name;

                autd.Open(LinkType.SOEM, ifname);
                // If you have already recognized the EtherCAT adapter name, you can write it directly like below.
                //autd.Open(LinkType.SOEM, "\\Device\\NPF_{B5B631C6-ED16-4780-9C4C-3941AE8120A6}");

                // AM
                Console.WriteLine("Amplitude Modulation");

                var mod = AUTD.SineModulation(150); // AM sin 150 Hz
                autd.AppendModulationSync(mod);

                var gain = AUTD.FocalPointGain(x, y, z); // Focal point @ (x, y, z) [mm]
                autd.AppendGainSync(gain);

                Console.ReadKey(true);

                // LM
                Console.WriteLine("Lateral Modulation");
                autd.AppendModulationSync(AUTD.Modulation(255));

                Gain f1 = AUTD.FocalPointGain(x + 2.5f, y, z);
                Gain f2 = AUTD.FocalPointGain(x - 2.5f, y, z);

                autd.AppendLateralGain(f1);
                autd.AppendLateralGain(f2);
                autd.StartLateralModulation(100); // Tapping f1 and f2 at 100Hz
           
                Console.WriteLine("press any key to finish...");
                Console.ReadKey(true);
            }
        }
    }
}
