using AUTD3Sharp;
using System;

namespace AUTD3SharpTest.Test
{
    class LateralTest
    {
        public static void Test()
        {
            Console.WriteLine("Start LateralModulation Test");

            float x = 83.0f;
            float y = 66.0f;
            float z = 150.0f;

            using (var autd = new AUTD())
            {
                autd.Open();
                autd.AddDevice(Vector3f.Zero, Vector3f.Zero);

                autd.AppendModulationSync(AUTD.Modulation(255));

                var f1 = AUTD.FocalPointGain(x - 7, y, z);
                var f2 = AUTD.FocalPointGain(x + 7, y, z);

                autd.AppendLateralGain(f1);
                autd.AppendLateralGain(f2);
                autd.StartLateralModulation(50);

                Console.WriteLine("press any key to finish...");
                Console.ReadKey(true);
            }
        }

        private LateralTest() { }
    }
}
