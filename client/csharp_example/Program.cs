/*
 *  Program.cs
 *  AUTD3SharpSample
 *
 *  Created by Shun Suzuki on 08/25/2019.
 *  Copyright © 2019 Hapis Lab. All rights reserved.
 *
 */

using AUTD3SharpTest.Test;
using System.Reflection;

[assembly: AssemblyVersion("3.3.1.0")]
namespace AUTD3SharpTest
{
    class Program
    {
        static void Main()
        {
            //SimpleExample.Test();
            //BesselExample.Test();
            //HoloGainExample.Test();
            //LateralExmaple.Test();

            SimpleExample_SOEM.Test();

            // Following test needs 2 AUTDs
            //GroupedGainTest.Test();
        }
    }
}