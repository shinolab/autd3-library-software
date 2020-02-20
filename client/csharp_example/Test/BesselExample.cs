/*
 * File: BesselExample.cs
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
    internal class BesselExample
    {
        public static void Test()
        {
            Console.WriteLine("Start BesselBeam Test");

            double x = AUTD.AUTDWidth / 2;
            double y = AUTD.AUTDHeight / 2;

            using (AUTD autd = new AUTD())
            {
                autd.Open();
                autd.AddDevice(Vector3d.Zero, Vector3d.Zero);

                autd.AppendModulationSync(AUTD.SineModulation(150)); // AM sin 150 HZ

                Vector3d start = new Vector3d(x, y, 0);
                Vector3d dir = Vector3d.UnitZ;
                autd.AppendGainSync(AUTD.BesselBeamGain(start, dir, 13.0 / 180 * AUTD.Pi)); // BesselBeam from (x, y, 0), theta = 13 deg

                Console.WriteLine("press any key to finish...");
                Console.ReadKey(true);
            }
        }
    }
}
