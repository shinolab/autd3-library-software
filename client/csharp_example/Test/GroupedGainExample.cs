/*
 * File: GroupedGainExample.cs
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
    internal class GroupedGainTest
    {
        public static void Test()
        {
            Console.WriteLine("Start GroupedGain Test");

            double x = 83.0;
            double y = 66.0;
            double z = 150.0;

            using (AUTD autd = new AUTD())
            {
                autd.Open();
                autd.AddDevice(Vector3d.Zero, Vector3d.Zero, 0); // Group ID 0
                autd.AddDevice(Vector3d.Zero, Vector3d.Zero, 1); // Group ID 1

                autd.AppendModulationSync(AUTD.SineModulation(150)); // AM sin 150 HZ

                autd.AppendGainSync(AUTD.GroupedGain(
                    new GainPair(0, AUTD.FocalPointGain(x, y, z))  // ID 0 : FocalPoint
                    , new GainPair(1, AUTD.BesselBeamGain(x, y, 0, 0, 0, 1, 13.0 * AUTD.Pi / 180)) // ID 1 : BesselBeam
                    ));

                Console.WriteLine("press any key to finish...");
                Console.ReadKey(true);
            }
        }

        private GroupedGainTest() { }
    }
}
