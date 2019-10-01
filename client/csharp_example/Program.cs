/*
 * File: Program.cs
 * Project: csharp_example
 * Created Date: 25/08/2019
 * Author: Shun Suzuki
 * -----
 * Last Modified: 26/09/2019
 * Modified By: Shun Suzuki (suzuki@hapis.k.u-tokyo.ac.jp)
 * -----
 * Copyright (c) 2019 Hapis Lab. All rights reserved.
 * 
 */

using AUTD3SharpTest.Test;
using System.Reflection;
using System.Threading;

namespace AUTD3SharpTest
{
    internal class Program
    {
        private static void Main()
        {
            //// Following samples require TwinCAT
            //SimpleExample.Test();
            //BesselExample.Test();
            //HoloGainExample.Test();
            //LateralExmaple.Test();

            // This don't need TwinCAT
            SimpleExample_SOEM.Test();
    
            // Following test needs 2 AUTDs
            //GroupedGainTest.Test();
        }
    }
}
