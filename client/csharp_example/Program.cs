/*
 * File: Program.cs
 * Project: csharp_example
 * Created Date: 25/08/2019
 * Author: Shun Suzuki
 * -----
 * Last Modified: 04/09/2019
 * Modified By: Shun Suzuki (suzuki@hapis.k.u-tokyo.ac.jp)
 * -----
 * Copyright (c) 2019 Hapis Lab. All rights reserved.
 * 
 */

using AUTD3SharpTest.Test;
using System.Reflection;

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