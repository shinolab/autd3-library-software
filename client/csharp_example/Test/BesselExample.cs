/*
 *  BesselExample.cs
 *  AUTD3SharpSample
 *
 *  Created by Shun Suzuki on 08/25/2019.
 *  Copyright © 2019 Hapis Lab. All rights reserved.
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

            var x = 83.0f;
            var y = 66.0f;

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
