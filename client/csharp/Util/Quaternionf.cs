/*
*
*  Quaternionf.cs
*  AUTD3Sharp
*  
*  Created by Shun Suzuki on 02/07/2018.
*  Copyright © 2018-2019 Hapis Lab. All rights reserved.
*
*/

using System;

namespace AUTD3Sharp
{
    public struct Quaternionf : IEquatable<Quaternionf>
    {
        #region ctor
        public Quaternionf(float x, float y, float z, float w)
        {
            X = x;
            Y = y;
            Z = z;
            W = w;
        }
        #endregion

        #region property
        public float X { get; }
        public float Y { get; }
        public float Z { get; }
        public float W { get; }
        #endregion

        #region indexcer
        public float this[int index]
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
        public static bool operator ==(Quaternionf left, Quaternionf right) => left.Equals(right);
        public static bool operator !=(Quaternionf left, Quaternionf right) => !left.Equals(right);

        public bool Equals(Quaternionf other)
        {
            return X.Equals(other.X) && Y.Equals(other.Y) && Z.Equals(other.Z) && W.Equals(other.W);
        }

        public override bool Equals(object obj)
        {
            if (obj is Quaternionf qua)
                return Equals(qua);
            else return false;
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
