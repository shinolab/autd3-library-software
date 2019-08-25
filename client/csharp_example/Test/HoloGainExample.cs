/*
 *  HoloGainExample.cs
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
    class HoloGainExample
    {
        public static void Test()
        {
            Console.WriteLine("Start HoloGain Test");

            float x = 83.0f;
            float y = 66.0f;
            float z = 150.0f;

            using (var autd = new AUTD())
            {
                autd.Open();
                autd.AddDevice(Vector3f.Zero, Vector3f.Zero);
                autd.AppendModulationSync(AUTD.SineModulation(150)); // AM sin 150 HZ

                var focuses = new[] {
                    new Vector3f(x - 30, y ,z),
                    new Vector3f(x + 30, y ,z)
                };
                var amps = new[] {
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
