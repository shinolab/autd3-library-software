using AUTD3Sharp;
using System;

namespace AUTD3SharpTest.Test
{
    class GroupedGainTest
    {
        public static void Test()
        {
            Console.WriteLine("Start GroupedGain Test");

            float x = 83.0f;
            float y = 66.0f;
            float z = 150.0f;

            using (var autd = new AUTD())
            {
                autd.Open();
                autd.AddDevice(Vector3f.Zero, Vector3f.Zero, 0); // Group ID 0
                autd.AddDevice(Vector3f.Zero, Vector3f.Zero, 1); // Group ID 1

                autd.AppendModulationSync(AUTD.SineModulation(150)); // AM sin 150 HZ

                autd.AppendGainSync(AUTD.GroupedGain(
                    new GainPair(0, AUTD.FocalPointGain(x, y, z))  // ID 0 : FocalPoint
                    , new GainPair(1, AUTD.BesselBeamGain(x, y, 0, 0, 0, 1, 13.0f * AUTD.Pi / 180)) // ID 1 : BesselBeam
                    ));

                Console.WriteLine("press any key to finish...");
                Console.ReadKey(true);
            }
        }

        private GroupedGainTest() { }
    }
}
