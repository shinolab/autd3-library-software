/*
 * File: Vector3d.cs
 * Project: Util
 * Created Date: 02/07/2018
 * Author: Shun Suzuki
 * -----
 * Last Modified: 20/02/2020
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
    public struct Vector3d : IEquatable<Vector3d>, IEnumerable<double>
    {
        #region ctor
        public Vector3d(double x, double y, double z)
        {
            X = x;
            Y = y;
            Z = z;
        }

        public Vector3d(params double[] vector)
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
        public static Vector3d UnitX => new Vector3d(1, 0, 0);
        public static Vector3d UnitY => new Vector3d(0, 1, 0);
        public static Vector3d UnitZ => new Vector3d(0, 0, 1);
        public static Vector3d Zero => new Vector3d(0, 0, 0);
        public Vector3d Normalized => this / L2Norm;
        public double L2Norm => Math.Sqrt(L2NormSquared);
        public double L2NormSquared => X * X + Y * Y + Z * Z;
        public double X { get; }
        public double Y { get; }
        public double Z { get; }
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
                    default: throw new ArgumentOutOfRangeException(nameof(index));
                }
            }
        }
        #endregion

        #region arithmetic
        public static Vector3d Negate(Vector3d operand)
        {
            return new Vector3d(-operand.X, -operand.Y, -operand.Z);
        }

        public static Vector3d Add(Vector3d left, Vector3d right)
        {
            double v1 = left.X + right.X;
            double v2 = left.Y + right.Y;
            double v3 = left.Z + right.Z;
            return new Vector3d(v1, v2, v3);
        }
        public static Vector3d Subtract(Vector3d left, Vector3d right)
        {
            double v1 = left.X - right.X;
            double v2 = left.Y - right.Y;
            double v3 = left.Z - right.Z;
            return new Vector3d(v1, v2, v3);
        }
        public static Vector3d Divide(Vector3d left, double right)
        {
            double v1 = left.X / right;
            double v2 = left.Y / right;
            double v3 = left.Z / right;

            return new Vector3d(v1, v2, v3);
        }
        public static Vector3d Multiply(Vector3d left, double right)
        {
            double v1 = left.X * right;
            double v2 = left.Y * right;
            double v3 = left.Z * right;
            return new Vector3d(v1, v2, v3);
        }

        public static Vector3d Multiply(double left, Vector3d right)
        {
            return Multiply(right, left);
        }

        public static Vector3d operator -(Vector3d operand)
        {
            return Negate(operand);
        }

        public static Vector3d operator +(Vector3d left, Vector3d right)
        {
            return Add(left, right);
        }

        public static Vector3d operator -(Vector3d left, Vector3d right)
        {
            return Subtract(left, right);
        }

        public static Vector3d operator *(Vector3d left, double right)
        {
            return Multiply(left, right);
        }

        public static Vector3d operator *(double left, Vector3d right)
        {
            return Multiply(right, left);
        }

        public static Vector3d operator /(Vector3d left, double right)
        {
            return Divide(left, right);
        }

        public static bool operator ==(Vector3d left, Vector3d right)
        {
            return left.Equals(right);
        }

        public static bool operator !=(Vector3d left, Vector3d right)
        {
            return !left.Equals(right);
        }

        public bool Equals(Vector3d other)
        {
            return X.Equals(other.X) && Y.Equals(other.Y) && Z.Equals(other.Z);
        }

        public override bool Equals(object obj)
        {
            if (obj is Vector3d vec)
            {
                return Equals(vec);
            }

            return false;
        }
        #endregion

        #region public methods
        public Vector3d Rectify()
        {
            return new Vector3d(Math.Max(X, 0), Math.Max(Y, 0), Math.Max(Z, 0));
        }

        public double[] ToArray()
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

        public IEnumerator<double> GetEnumerator()
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
