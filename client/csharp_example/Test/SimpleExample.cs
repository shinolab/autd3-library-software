using AUTD3Sharp;
using System;

namespace AUTD3SharpTest.Test
{
    class SimpleExample
    {
        public static void Test()
        {
            Console.WriteLine("Start Simple Test");

            float x = 83.0f;
            float y = 66.0f;
            float z = 150.0f;

            using (var autd = new AUTD())
            {
                autd.Open();
                autd.AddDevice(Vector3f.Zero, Vector3f.Zero);
                autd.SetSilentMode(true);
                autd.SetLMSilentMode(false);

                //autd.AppendModulationSync(AUTD.SineModulation(200)); // AM sin 150 HZ
                autd.AppendModulationSync(AUTD.SineModulation(150)); // AM sin 150 HZ

                var gain = AUTD.FocalPointGain(x, y, z);
                //gain.Fix(); // 位相振幅独立制御用の修正

                autd.AppendGainSync(gain); // FocalPoint at (x,y,z)

                Console.WriteLine("SILENT");
                Console.ReadKey(true);

                autd.SetSilentMode(false);
                autd.SetLMSilentMode(false);
                autd.AppendGainSync(gain); // FocalPoint at (x,y,z)

                Console.WriteLine("NO SILENT");
                Console.WriteLine("press any key to finish...");
                Console.ReadKey(true);
            }
        }

        private SimpleExample() { }
    }
}
