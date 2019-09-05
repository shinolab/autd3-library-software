/*
 * File: Vector3f.cs
 * Project: Util
 * Created Date: 02/07/2018
 * Author: Shun Suzuki
 * -----
 * Last Modified: 05/09/2019
 * Modified By: Shun Suzuki (suzuki@hapis.k.u-tokyo.ac.jp)
 * -----
 * Copyright (c) 2018-2019 Hapis Lab. All rights reserved.
 * 
 */

using System;
using System.Collections;
using System.Collections.Generic;
using System.Globalization;

namespace AUTD3Sharp
{
    // ReSharper disable once InconsistentNaming
    public struct Vector3f : IEquatable<Vector3f>, IEnumerable<float>
    {
        #region ctor
        public Vector3f(float x, float y, float z)
        {
            X = x;
            Y = y;
            Z = z;
        }

        public Vector3f(params float[] vector)
        {
            if (vector == null)
            {
                throw new ArgumentNullException(nameof(vector));
            }

            if (vector.Length != 3)
            {
                throw new InvalidCastException();
            }

            X = vector[0];
            Y = vector[1];
            Z = vector[2];
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
        public float X { get; }
        public float Y { get; }
        public float Z { get; }
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
                }
            }
        }
        #endregion

        #region arithmetic
        public static Vector3f Negate(Vector3f operand)
        {
            return new Vector3f(-operand.X, -operand.Y, -operand.Z);
        }

        public static Vector3f Add(Vector3f left, Vector3f right)
        {
            float v1 = left.X + right.X;
            float v2 = left.Y + right.Y;
            float v3 = left.Z + right.Z;
            return new Vector3f(v1, v2, v3);
        }
        public static Vector3f Subtract(Vector3f left, Vector3f right)
        {
            float v1 = left.X - right.X;
            float v2 = left.Y - right.Y;
            float v3 = left.Z - right.Z;
            return new Vector3f(v1, v2, v3);
        }
        public static Vector3f Divide(Vector3f left, float right)
        {
            float v1 = left.X / right;
            float v2 = left.Y / right;
            float v3 = left.Z / right;

            return new Vector3f(v1, v2, v3);
        }
        public static Vector3f Multiply(Vector3f left, float right)
        {
            float v1 = left.X * right;
            float v2 = left.Y * right;
            float v3 = left.Z * right;
            return new Vector3f(v1, v2, v3);
        }

        public static Vector3f Multiply(float left, Vector3f right)
        {
            return Multiply(right, left);
        }

        public static Vector3f operator -(Vector3f operand)
        {
            return Negate(operand);
        }

        public static Vector3f operator +(Vector3f left, Vector3f right)
        {
            return Add(left, right);
        }

        public static Vector3f operator -(Vector3f left, Vector3f right)
        {
            return Subtract(left, right);
        }

        public static Vector3f operator *(Vector3f left, float right)
        {
            return Multiply(left, right);
        }

        public static Vector3f operator *(float left, Vector3f right)
        {
            return Multiply(right, left);
        }

        public static Vector3f operator /(Vector3f left, float right)
        {
            return Divide(left, right);
        }

        public static bool operator ==(Vector3f left, Vector3f right)
        {
            return left.Equals(right);
        }

        public static bool operator !=(Vector3f left, Vector3f right)
        {
            return !left.Equals(right);
        }

        public bool Equals(Vector3f other)
        {
            return X.Equals(other.X) && Y.Equals(other.Y) && Z.Equals(other.Z);
        }

        public override bool Equals(object obj)
        {
            if (obj is Vector3f vec)
            {
                return Equals(vec);
            }

            return false;
        }
        #endregion

        #region public methods
        public Vector3f Rectify()
        {
            return new Vector3f(Math.Max(X, 0), Math.Max(Y, 0), Math.Max(Z, 0));
        }

        public float[] ToArray()
        {
            return new[] { X, Y, Z };
        }
        #endregion

        #region util
        public override int GetHashCode()
        {
            return X.GetHashCode() ^ Y.GetHashCode() ^ Z.GetHashCode();
        }

        IEnumerator IEnumerable.GetEnumerator()
        {
            return GetEnumerator();
        }

        public string ToString(string format)
        {
            return "3d Column Vector:\n"
+ string.Format(CultureInfo.CurrentCulture, format, X) + "\n"
+ string.Format(CultureInfo.CurrentCulture, format, Y) + "\n"
+ string.Format(CultureInfo.CurrentCulture, format, Z);
        }

        public IEnumerator<float> GetEnumerator()
        {
            yield return X;
            yield return Y;
            yield return Z;
        }

        public override string ToString()
        {
            return ToString("{0,-20}");
        }
        #endregion
    }
}
