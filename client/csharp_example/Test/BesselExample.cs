/*
 * File: BesselExample.cs
 * Project: Test
 * Created Date: 25/08/2019
 * Author: Shun Suzuki
 * -----
 * Last Modified: 05/09/2019
 * Modified By: Shun Suzuki (suzuki@hapis.k.u-tokyo.ac.jp)
 * -----
 * Copyright (c) 2019 Hapis Lab. All rights reserved.
 * 
 */

using AUTD3Sharp;
using System;

namespace AUTD3SharpTest.Test
{
    class BesselExample
    {
        public static void Test()
        {
            Console.WriteLine("Start BesselBeam Test");

            float x = AUTD.AUTDWidth / 2;
            float y = AUTD.AUTDHeight / 2;

            using (var autd = new AUTD())
            {
                autd.Open();
                autd.AddDevice(Vector3f.Zero, Vector3f.Zero);

                autd.AppendModulationSync(AUTD.SineModulation(150)); // AM sin 150 HZ

                var start = new Vector3f(x, y, 0);
                var dir = Vector3f.UnitZ;
                autd.AppendGainSync(AUTD.BesselBeamGain(start, dir, 13.0f / 180 * AUTD.Pi)); // BesselBeam from (x, y, 0), theta = 13 deg

                Console.WriteLine("press any key to finish...");
                Console.ReadKey(true);
            }
        }
    }
}
