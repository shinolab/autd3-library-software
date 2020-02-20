/*
 * File: SimpleExample.cs
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

namespace AUTD3SharpTest.Test
{
    internal class SimpleExample
    {
        public static void Test()
        {
            Console.WriteLine("Start Simple Test");

            double x = AUTD.AUTDWidth / 2;
            double y = AUTD.AUTDHeight / 2;
            double z = 150;

            using (AUTD autd = new AUTD())
            {
                autd.Open();
                autd.AddDevice(Vector3d.Zero, Vector3d.Zero);

                Modulation mod = AUTD.SineModulation(150); // AM sin 150 Hz
                autd.AppendModulationSync(mod);

                Gain gain = AUTD.FocalPointGain(x, y, z); // Focal point @ (x, y, z) [mm]
                autd.AppendGainSync(gain);

                Console.WriteLine("press any key to finish...");
                Console.ReadKey(true);
            }
        }
    }
}
