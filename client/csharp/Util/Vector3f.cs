/*
*
*  Created by Shun Suzuki on 02/07/2018.
*
*/

using System;
using System.Globalization;

namespace AUTD3Sharp
{
    public struct Vector3f : IEquatable<Vector3f>
    {
        #region ctor
        public Vector3f(float x, float y, float z)
        {
            this.X = x;
            this.Y = y;
            this.Z = z;
        }
        #endregion

        #region property
        public static Vector3f UnitX => new Vector3f(1, 0, 0);
        public static Vector3f UnitY => new Vector3f(0, 1, 0);
        public static Vector3f UnitZ => new Vector3f(0, 0, 1);
        public static Vector3f Zero => new Vector3f(0, 0, 0);
        public Vector3f Normalized => this / L2Norm;
        public float L2Norm => (float)Math.Sqrt(L2NormSquared);
        public float L2NormSquared => X * X + Y * Y + Z * Z;
        public float X { get; internal set; }
        public float Y { get; internal set; }
        public float Z { get; internal set; }
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
                    default: throw new ArgumentOutOfRangeException(nameof(index));
                };
            }
            set
            {
                switch (index)
                {
                    case 0: X = value; break;
                    case 1: Y = value; break;
                    case 2: Z = value; break;
                    default: throw new ArgumentOutOfRangeException(nameof(index));
                };
            }
        }
        #endregion

        #region arithmetic
        public static Vector3f Negate(Vector3f operand) => new Vector3f(-operand.X, -operand.Y, -operand.Z);
        public static Vector3f Add(Vector3f left, Vector3f right)
        {
            var v1 = left.X + right.X;
            var v2 = left.Y + right.Y;
            var v3 = left.Z + right.Z;
            return new Vector3f(v1, v2, v3);
        }
        public static Vector3f Subtract(Vector3f left, Vector3f right)
        {
            var v1 = left.X - right.X;
            var v2 = left.Y - right.Y;
            var v3 = left.Z - right.Z;
            return new Vector3f(v1, v2, v3);
        }
        public static Vector3f Divide(Vector3f left, float right)
        {
            var v1 = left.X / right;
            var v2 = left.Y / right;
            var v3 = left.Z / right;

            return new Vector3f(v1, v2, v3);
        }
        public static Vector3f Multiply(Vector3f left, float right)
        {
            var v1 = left.X * right;
            var v2 = left.Y * right;
            var v3 = left.Z * right;
            return new Vector3f(v1, v2, v3);
        }

        public static Vector3f Multiply(float left, Vector3f right) => Multiply(right, left);

        public static Vector3f operator -(Vector3f operand) => Negate(operand);
        public static Vector3f operator +(Vector3f left, Vector3f right) => Add(left, right);
        public static Vector3f operator -(Vector3f left, Vector3f right) => Subtract(left, right);
        public static Vector3f operator *(Vector3f left, float right) => Multiply(left, right);
        public static Vector3f operator *(float left, Vector3f right) => Multiply(right, left);
        public static Vector3f operator /(Vector3f left, float right) => Divide(left, right);

        public static bool operator ==(Vector3f left, Vector3f right) => left.Equals(right);
        public static bool operator !=(Vector3f left, Vector3f right) => !left.Equals(right);

        public bool Equals(Vector3f other)
        {
            return this.X == other.X && this.Y == other.Y && this.Z == other.Z;
        }

        public override bool Equals(object obj)
        {
            if (obj is Vector3f vec)
                return this.Equals(vec);
            else return false;
        }
        #endregion

        #region public methods
        public void Rectify()
        {
            X = Math.Max(X, 0);
            Y = Math.Max(Y, 0);
            Z = Math.Max(Z, 0);
        }

        public float[] ToArray() => new float[] { X, Y, Z };
        #endregion

        #region util
        public override int GetHashCode()
        {
            return this.X.GetHashCode() ^ this.Y.GetHashCode() ^ this.Z.GetHashCode();
        }
        public string ToString(string format) => "3d Column Vector:\n"
                + String.Format(CultureInfo.CurrentCulture, format, X) + "\n"
                + String.Format(CultureInfo.CurrentCulture, format, Y) + "\n"
                + String.Format(CultureInfo.CurrentCulture, format, Z);
        public override string ToString() => ToString("{0,-20}");
        #endregion
    }
}
