/*
 * File: Quaterniond.cs
 * Project: Util
 * Created Date: 02/07/2018
 * Author: Shun Suzuki
 * -----
 * Last Modified: 20/02/2020
 * Modified By: Shun Suzuki (suzuki@hapis.k.u-tokyo.ac.jp)
 * -----
 * Copyright (c) 2019 Hapis Lab. All rights reserved.
 * 
 */

using System;

namespace AUTD3Sharp
{
    public struct Quaterniond : IEquatable<Quaterniond>
    {
        #region ctor
        public Quaterniond(double x, double y, double z, double w)
        {
            X = x;
            Y = y;
            Z = z;
            W = w;
        }
        #endregion

        #region property
        public double X { get; }
        public double Y { get; }
        public double Z { get; }
        public double W { get; }
        #endregion

        #region indexcer
        public double this[int index]
        {
            get
            {
                switch (index)
                {
                    case 0: return X;
                    case 1: return Y;
                    case 2: return Z;
                    case 3: return W;
                    default: throw new ArgumentOutOfRangeException(nameof(index));
                }
            }
        }
        #endregion


        #region arithmetic
        public static bool operator ==(Quaterniond left, Quaterniond right)
        {
            return left.Equals(right);
        }

        public static bool operator !=(Quaterniond left, Quaterniond right)
        {
            return !left.Equals(right);
        }

        public bool Equals(Quaterniond other)
        {
            return X.Equals(other.X) && Y.Equals(other.Y) && Z.Equals(other.Z) && W.Equals(other.W);
        }

        public override bool Equals(object obj)
        {
            if (obj is Quaterniond qua)
            {
                return Equals(qua);
            }
            else
            {
                return false;
            }
        }
        #endregion

        #region util
        public override int GetHashCode()
        {
            return X.GetHashCode() ^ Y.GetHashCode() ^ Z.GetHashCode() ^ W.GetHashCode();
        }
        #endregion
    }
}
