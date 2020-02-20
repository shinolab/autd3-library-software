/*
 * File: LateralExample.cs
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
    internal class STMExmaple
    {
        public static void Test()
        {
            Console.WriteLine("Start Spatio-Temporal Modulation Test");

            double x = 83.0;
            double y = 66.0;
            double z = 150.0;

            using (AUTD autd = new AUTD())
            {
                autd.Open();
                autd.AddDevice(Vector3d.Zero, Vector3d.Zero);

                autd.SetSilentMode(false);

                autd.AppendModulationSync(AUTD.Modulation(255));

                Gain f1 = AUTD.FocalPointGain(x + 10, y, z);
                Gain f2 = AUTD.FocalPointGain(x - 10, y, z);

                autd.AppendSTMGain(f1);
                autd.AppendSTMGain(f2);
                autd.StartSTModulation(50);

                Console.WriteLine("press any key to finish...");
                Console.ReadKey(true);
            }
        }
    }
}
