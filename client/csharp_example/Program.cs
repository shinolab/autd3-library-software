using AUTD3SharpTest.Test;

namespace AUTD3SharpTest
{
    class Program
    {
        static void Main()
        {
            SimpleTest.Test();
            BesselTest.Test();
            HoloGainTest.Test();
            LateralTest.Test();

            // Following test needs 2 AUTDs
            //GroupedGainTest.Test();
        }
    }
}