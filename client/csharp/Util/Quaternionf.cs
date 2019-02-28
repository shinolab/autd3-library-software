/*
*
*  Created by Shun Suzuki on 02/07/2018.
*
*/

using System;
using System.Globalization;

namespace AUTD3Sharp
{
    public struct Quaternionf : IEquatable<Quaternionf>
    {
        #region ctor
        public Quaternionf(float x, float y, float z, float w)
        {
            this.X = x;
            this.Y = y;
            this.Z = z;
            this.W = w;
        }
        #endregion

        #region property
        public float X { get; internal set; }
        public float Y { get; internal set; }
        public float Z { get; internal set; }
        public float W { get; internal set; }
        #endregion

        #region arithmetic
        public static bool operator ==(Quaternionf left, Quaternionf right) => left.Equals(right);
        public static bool operator !=(Quaternionf left, Quaternionf right) => !left.Equals(right);

        public bool Equals(Quaternionf other)
        {
            return this.X == other.X && this.Y == other.Y && this.Z == other.Z && this.W == other.W;
        }

        public override bool Equals(object obj)
        {
            if (obj is Quaternionf qua)
                return this.Equals(qua);
            else return false;
        }
        #endregion

        #region util
        public override int GetHashCode()
        {
            return this.X.GetHashCode() ^ this.Y.GetHashCode() ^ this.Z.GetHashCode() ^ this.W.GetHashCode();
        }
        #endregion
    }
}
