/*
 * File: HoloGainExample.cs
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
    internal class HoloGainExample
    {
        public static void Test()
        {
            Console.WriteLine("Start HoloGain Test");

            float x = 83.0f;
            float y = 66.0f;
            float z = 150.0f;

            using (AUTD autd = new AUTD())
            {
                autd.Open();
                autd.AddDevice(Vector3f.Zero, Vector3f.Zero);
                autd.AppendModulationSync(AUTD.SineModulation(150)); // AM sin 150 HZ

                Vector3f[] focuses = new[] {
                    new Vector3f(x - 30, y ,z),
                    new Vector3f(x + 30, y ,z)
                };
                float[] amps = new[] {
                    1.0f,
                    1.0f
                };
                autd.AppendGainSync(AUTD.HoloGain(focuses, amps));

                Console.WriteLine("press any key to finish...");
                Console.ReadKey(true);
            }
        }
    }
}
