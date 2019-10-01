/*
 * File: GainMap.cs
 * Project: Util
 * Created Date: 07/10/2018
 * Author: Shun Suzuki
 * -----
 * Last Modified: 05/09/2019
 * Modified By: Shun Suzuki (suzuki@hapis.k.u-tokyo.ac.jp)
 * -----
 * Copyright (c) 2019 Hapis Lab. All rights reserved.
 * 
 */

using System;
using System.Linq;

namespace AUTD3Sharp
{
    public class GainMap
    {
        public int Size { get; }

        private readonly int[] _ids;
        private readonly IntPtr[] _gains;

        public unsafe IntPtr* GainPointer
        {
            get
            {
                fixed (IntPtr* p = _gains)
                {
                    return p;
                }
            }
        }
        public unsafe int* IdPointer
        {
            get
            {
                fixed (int* p = _ids)
                {
                    return p;
                }
            }
        }

        public GainMap(params GainPair[] gainPairs)
        {
            if (gainPairs == null)
            {
                throw new ArgumentNullException(nameof(gainPairs));
            }

            Size = gainPairs.Length;
            _ids = new int[Size];
            _gains = new IntPtr[Size];
            for (int i = 0; i < Size; i++)
            {
                _ids[i] = gainPairs[i].Id;
                _gains[i] = gainPairs[i].Gain.GainPtr;
            }

            bool duplication = Size > _ids.GroupBy(i => i).Count();
            if (duplication)
            {
                throw new ArgumentException("Multiple Gains are set for the same Group ID");
            }
        }
    }

    public struct GainPair : IEquatable<GainPair>
    {
        public int Id { get; }
        public Gain Gain { get; }
        public GainPair(int id, Gain gain)
        {
            Id = id;
            Gain = gain;
        }

        public static bool operator ==(GainPair left, GainPair right)
        {
            return left.Equals(right);
        }

        public static bool operator !=(GainPair left, GainPair right)
        {
            return !left.Equals(right);
        }

        public bool Equals(GainPair other)
        {
            return Id == other.Id && Gain == other.Gain;
        }

        public override bool Equals(object obj)
        {
            if (obj is GainPair pair)
            {
                return Equals(pair);
            }

            return false;
        }
        public override int GetHashCode()
        {
            return Id ^ Gain.GetHashCode();
        }
    }
}
