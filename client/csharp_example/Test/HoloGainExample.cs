/*
 * File: HoloGainExample.cs
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
    internal class HoloGainExample
    {
        public static void Test()
        {
            Console.WriteLine("Start HoloGain Test");

            double x = 83.0;
            double y = 66.0;
            double z = 150.0;

            using (AUTD autd = new AUTD())
            {
                autd.Open();
                autd.AddDevice(Vector3d.Zero, Vector3d.Zero);
                autd.AppendModulationSync(AUTD.SineModulation(150)); // AM sin 150 HZ

                Vector3d[] focuses = new[] {
                    new Vector3d(x - 30, y ,z),
                    new Vector3d(x + 30, y ,z)
                };
                double[] amps = new[] {
                    1.0,
                    1.0
                };
                autd.AppendGainSync(AUTD.HoloGain(focuses, amps));

                Console.WriteLine("press any key to finish...");
                Console.ReadKey(true);
            }
        }
    }
}
