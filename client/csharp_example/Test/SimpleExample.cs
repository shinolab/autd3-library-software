/*
 *  SimpleExample.cpp
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
    class SimpleExample
    {
        public static void Test()
        {
            Console.WriteLine("Start Simple Test");

            float x = 90.0f;
            float y = 151.4f;
            float z = 215.0f;

            using (var autd = new AUTD())
            {
                autd.Open();
                autd.AddDevice(Vector3f.Zero, Vector3f.Zero);
                autd.AddDevice(Vector3f.UnitY * AUTD.AUTDHeight, Vector3f.Zero);

                var mod = AUTD.SineModulation(150); // AM sin 150 Hz
                autd.AppendModulationSync(mod);

                var gain = AUTD.FocalPointGain(x, y, z); // Focal point @ (x, y, z) [mm]
                autd.AppendGainSync(gain);

                Console.WriteLine("press any key to finish...");
                Console.ReadKey(true);
            }
        }
    }
}
