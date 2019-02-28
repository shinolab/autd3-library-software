/*
*
*  Created by Shun Suzuki on 10/07/2018.
*  Copyright © 2018 Hapis Lab. All rights reserved.
*
*/

using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;

namespace AUTD3Sharp
{
    public class GainMap 
    {
        public int Size { get; private set; }

        private readonly int[] _IDs;
        private readonly IntPtr[] _gains;

        public unsafe IntPtr* GainPointer
        {
            get {
                fixed (IntPtr* p = _gains) return p;
            }
        }
        public unsafe int* IdPointer {
            get
            {
                fixed (int* p = _IDs) return p;
            }
        }

        public GainMap(params GainPair[] gainPairs)
        {
            if (gainPairs == null) throw new ArgumentNullException(nameof(gainPairs));

            this.Size = gainPairs.Length;
            this._IDs = new int[this.Size];
            this._gains = new IntPtr[this.Size];
            for (int i = 0; i < this.Size; i++)
            {
                this._IDs[i] = gainPairs[i].Id;
                this._gains[i] = gainPairs[i].Gain.GainPtr;
            }

            var duplication = this.Size > this._IDs.GroupBy(i => i).Count();
            if (duplication)
                throw new ArgumentException("Multiple Gains are set for the same Group ID");
        }
    }

    public struct GainPair
    {
        public int Id { get; private set; }
        public Gain Gain { get; private set; }
        public GainPair(int id, Gain gain)
        {
            this.Id = id;
            this.Gain = gain;
        }

        public static bool operator ==(GainPair left, GainPair right) => left.Equals(right);
        public static bool operator !=(GainPair left, GainPair right) => !left.Equals(right);
        public bool Equals(GainPair other) => this.Id == other.Id && this.Gain == other.Gain;

        public override bool Equals(object obj)
        {
            if (obj is GainPair pair)
                return this.Equals(pair);
            return false;
        }
        public override int GetHashCode() => Id ^ Gain.GetHashCode();

    }
}
