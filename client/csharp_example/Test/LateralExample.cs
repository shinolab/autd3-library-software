using AUTD3Sharp;
using System;

namespace AUTD3SharpTest.Test
{
    class LateralExmaple
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

                autd.SetSilentMode(false);
                autd.SetLMSilentMode(false);

                autd.AppendModulationSync(AUTD.Modulation(255));

                var f1 = AUTD.FocalPointGain(x + 10, y, z);
                var f2 = AUTD.FocalPointGain(x - 10, y, z);

                autd.AppendLateralGain(f1);
                autd.AppendLateralGain(f2);
                autd.StartLateralModulation(50);

                Console.WriteLine("press any key to finish...");
                Console.ReadKey(true);

                autd.FinishLateralModulation();
                autd.ResetLateralGain();

                autd.SetSilentMode(true);
                autd.SetLMSilentMode(false);

                autd.AppendLateralGain(f1);
                autd.AppendLateralGain(f2);
                autd.StartLateralModulation(50);

                Console.WriteLine("press any key to finish...");
                Console.ReadKey(true);

                autd.FinishLateralModulation();
                autd.ResetLateralGain();

                autd.SetSilentMode(false);
                autd.SetLMSilentMode(true);

                autd.AppendLateralGain(f1);
                autd.AppendLateralGain(f2);
                autd.StartLateralModulation(50);

                Console.WriteLine("press any key to finish...");
                Console.ReadKey(true);

                autd.FinishLateralModulation();
                autd.ResetLateralGain();
            }
        }

        private LateralExmaple() { }
    }
}
