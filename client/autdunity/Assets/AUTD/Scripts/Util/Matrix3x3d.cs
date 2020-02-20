/*
 * File: Matrix3x3d.cs
 * Project: Util
 * Created Date: 24/07/2018
 * Author: Shun Suzuki
 * -----
 * Last Modified: 20/02/2020
 * Modified By: Shun Suzuki (suzuki@hapis.k.u-tokyo.ac.jp)
 * -----
 * Copyright (c) 2019 Hapis Lab. All rights reserved.
 * 
 */

#if UNITY_2018_3_OR_NEWER
#define UNITY
#endif

using System;
using System.Collections;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;

#if UNITY
using UnityEngine;
using Vector3d = UnityEngine.Vector3;
#endif

namespace AUTD3Sharp
{
    // ReSharper disable once InconsistentNaming
    public unsafe struct Matrix3x3d : IEquatable<Matrix3x3d>, IEnumerable<double>
    {
        #region const
        private const double Eps = 1e-7;
        #endregion

        #region field
        public double M00 { get; }
        public double M01 { get; }
        public double M02 { get; }
        public double M10 { get; }
        public double M11 { get; }
        public double M12 { get; }
        public double M20 { get; }
        public double M21 { get; }
        public double M22 { get; }
        #endregion

        #region ctor
        public Matrix3x3d(params double[] mat)
        {
            if (mat == null)
            {
                throw new ArgumentNullException(nameof(mat));
            }

            if (mat.Length != 9)
            {
                throw new InvalidCastException();
            }

            M00 = mat[0];
            M01 = mat[1];
            M02 = mat[2];
            M10 = mat[3];
            M11 = mat[4];
            M12 = mat[5];
            M20 = mat[6];
            M21 = mat[7];
            M22 = mat[8];
        }

        public Matrix3x3d(Vector3d[] pcX, Vector3d[] pcY)
        {
            if (pcX == null)
            {
                throw new ArgumentNullException(nameof(pcX));
            }

            if (pcY == null)
            {
                throw new ArgumentNullException(nameof(pcY));
            }

            if (pcX.Length != pcY.Length)
            {
                throw new ArgumentException("Size of point sets must be the same.");
            }

            double[] mat = new double[9];

            fixed (Vector3d* xvp = pcX)
            fixed (Vector3d* yvp = pcY)
            {
                {
                    double* xp = (double*)xvp;
                    double* yp = (double*)yvp;

                    for (int i = 0; i < 3; i++)
                    {
                        for (int j = 0; j < 3; j++)
                        {
                            double temp = 0.0;
                            for (int k = 0; k < pcX.Length; k++)
                            {
                                temp += xp[k * 3 + i] * yp[k * 3 + j];
                            }

                            mat[i * 3 + j] = temp;
                        }
                    }
                }
            }

            M00 = mat[0];
            M01 = mat[1];
            M02 = mat[2];
            M10 = mat[3];
            M11 = mat[4];
            M12 = mat[5];
            M20 = mat[6];
            M21 = mat[7];
            M22 = mat[8];
        }

        public Matrix3x3d(Vector3d[] pcX, Vector3d[] pcY, double[] weight)
        {
            if (pcX == null)
            {
                throw new ArgumentNullException(nameof(pcX));
            }

            if (pcY == null)
            {
                throw new ArgumentNullException(nameof(pcY));
            }

            if (weight == null)
            {
                throw new ArgumentNullException(nameof(weight));
            }

            if (pcX.Length != pcY.Length || pcX.Length != weight.Length)
            {
                throw new ArgumentException("Size of point sets and weights must be the same.");
            }

            double[] mat = new double[9];

            fixed (Vector3d* xvp = pcX)
            fixed (Vector3d* yvp = pcY)
            fixed (double* wp = weight)
            {
                double* xp = (double*)xvp;
                double* yp = (double*)yvp;

                for (int i = 0; i < 3; i++)
                {
                    for (int j = 0; j < 3; j++)
                    {
                        double temp = 0.0;
                        for (int k = 0; k < pcX.Length; k++)
                        {
                            temp += xp[k * 3 + i] * yp[k * 3 + j] * wp[k];
                        }

                        mat[i * 3 + j] = temp;
                    }
                }
            }

            M00 = mat[0];
            M01 = mat[1];
            M02 = mat[2];
            M10 = mat[3];
            M11 = mat[4];
            M12 = mat[5];
            M20 = mat[6];
            M21 = mat[7];
            M22 = mat[8];
        }
        #endregion

        #region property
        public static Matrix3x3d Identity => CreateDiagonal(1.0, 1.0, 1.0);

        public double Determinant
        {
            get
            {
                double det = 0.0;

                det += M00 * M11 * M22 + M02 * M10 * M21 + M01 * M12 * M20;
                det -= M02 * M11 * M20 + M01 * M10 * M22 + M00 * M12 * M21;

                return det;
            }
        }
        #endregion

        #region indexcer
        public double this[int row, int col]
        {
            get
            {
                switch (row * 3 + col)
                {
                    case 0: return M00;
                    case 1: return M01;
                    case 2: return M02;

                    case 3: return M10;
                    case 4: return M11;
                    case 5: return M12;

                    case 6: return M20;
                    case 7: return M21;
                    case 8: return M22;

                    default:
                        if (2 < row)
                        {
                            throw new ArgumentOutOfRangeException(nameof(row));
                        }
                        else
                        {
                            throw new ArgumentOutOfRangeException(nameof(col));
                        }
                }
            }
        }
        #endregion

        #region factory
        public static Matrix3x3d CreateDiagonal(params double[] diagonal)
        {
            if (diagonal == null)
            {
                throw new ArgumentNullException(nameof(diagonal));
            }

            if (diagonal.Length != 3)
            {
                throw new InvalidCastException();
            }

            return new Matrix3x3d(diagonal[0], 0, 0, 0, diagonal[1], 0, 0, 0, diagonal[2]);
        }
        public static Matrix3x3d CreateDiagonal(Vector3d diagonal)
        {
            return CreateDiagonal(diagonal[0], diagonal[1], diagonal[2]);
        }

        public static Vector3d FromDiagonal(Matrix3x3d mat)
        {
            return new Vector3d(mat[0, 0], mat[1, 1], mat[2, 2]);
        }

        public static Matrix3x3d FromEulerXyz(double eulerX, double eulerY, double eulerZ)
        {
            Matrix3x3d xRot = new Matrix3x3d(1, 0, 0,
                                        0, Math.Cos(eulerX), -Math.Sin(eulerX),
                                        0, Math.Sin(eulerX), Math.Cos(eulerX));

            Matrix3x3d yRot = new Matrix3x3d(Math.Cos(eulerY), 0, Math.Sin(eulerY),
                                        0, 1, 0,
                                        -Math.Sin(eulerY), 0, Math.Cos(eulerY));

            Matrix3x3d zRot = new Matrix3x3d(Math.Cos(eulerZ), -Math.Sin(eulerZ), 0,
                                        Math.Sin(eulerZ), Math.Cos(eulerZ), 0,
                                        0, 0, 1);

            return zRot * yRot * xRot;
        }
        #endregion

        #region public methods
        /// <summary>
        /// 特異値計算
        /// </summary>
        /// <returns>結果を格納した構造体</returns>
        public SvdMatrices Svd()
        {
            Matrix3x3d x = this;

            Matrix3x3d xtx = x.Transpose() * x;
            Vector3d eigenvalue1 = Jacobi(xtx, out Matrix3x3d v);

            Matrix3x3d xxt = x * x.Transpose();
            Vector3d eigenvalue2 = Jacobi(xxt, out Matrix3x3d u);

            Vector3d eigenvalue = ((eigenvalue1 + eigenvalue2) / 2);
#if UNITY
            eigenvalue.x = eigenvalue.x > 0 ? eigenvalue.x: 0;
            eigenvalue.y = eigenvalue.y > 0 ? eigenvalue.y: 0;
            eigenvalue.z = eigenvalue.z > 0 ? eigenvalue.z: 0;
#else
            eigenvalue = eigenvalue.Rectify();
#endif
            Matrix3x3d s = CreateDiagonal(Math.Sqrt(eigenvalue[0]), Math.Sqrt(eigenvalue[1]), Math.Sqrt(eigenvalue[2]));

            Matrix3x3d sp = u.Transpose() * x * v;

            Matrix3x3d h = CreateDiagonal(new Vector3d(sp.M00 < 0 ? -1 : 1, sp.M11 < 0 ? -1 : 1, sp.M22 < 0 ? -1 : 1));
            v *= h;

            return new SvdMatrices(u, s, v);
        }

        /// <summary>
        /// Jacobi法による固有値および固有ベクトルの計算。
        /// 対称行列のみ適用可。
        /// </summary>
        /// <param name="symmetricMatrix">対称行列</param>
        /// <param name="eigenvectors">固有ベクトルの行列</param>
        /// <returns></returns>
        public static Vector3d Jacobi(Matrix3x3d symmetricMatrix, out Matrix3x3d eigenvectors)
        {
            double max;

            double[] eigenvectorsArray = new double[] { 1, 0, 0, 0, 1, 0, 0, 0, 1 };

            int count = 0;
            double[] eigenvalues = symmetricMatrix.ToArray();

            do
            {
                if (++count > 1000)
                {
                    break;
                }

                if (Math.Abs((max = GetMaxValue(eigenvalues, out int p, out int q))) < Eps)
                {
                    break;
                }

                double app = eigenvalues[p * 3 + p];
                double apq = eigenvalues[p * 3 + q];
                double aqq = eigenvalues[q * 3 + q];

                double alpha = (app - aqq) / 2;
                double beta = -apq;
                double gamma = Math.Abs(alpha) / Math.Sqrt(alpha * alpha + beta * beta);

                double s = Math.Sqrt((1 - gamma) / 2);
                double c = Math.Sqrt((1 + gamma) / 2);
                if (alpha * beta < 0)
                {
                    s = -s;
                }

                for (int i = 0; i < 3; i++)
                {
                    double temp = c * eigenvalues[p * 3 + i] - s * eigenvalues[q * 3 + i];
                    eigenvalues[q * 3 + i] = s * eigenvalues[p * 3 + i] + c * eigenvalues[q * 3 + i];
                    eigenvalues[p * 3 + i] = temp;
                }

                for (int i = 0; i < 3; i++)
                {
                    eigenvalues[i * 3 + p] = eigenvalues[p * 3 + i];
                    eigenvalues[i * 3 + q] = eigenvalues[q * 3 + i];
                }

                eigenvalues[p * 3 + p] = c * c * app + s * s * aqq - 2 * s * c * apq;
                eigenvalues[p * 3 + q] = s * c * (app - aqq) + (c * c - s * s) * apq;
                eigenvalues[q * 3 + p] = s * c * (app - aqq) + (c * c - s * s) * apq;
                eigenvalues[q * 3 + q] = s * s * app + c * c * aqq + 2 * s * c * apq;

                for (int i = 0; i < 3; i++)
                {
                    double temp = c * eigenvectorsArray[i * 3 + p] - s * eigenvectorsArray[i * 3 + q];
                    eigenvectorsArray[i * 3 + q] = s * eigenvectorsArray[i * 3 + p] + c * eigenvectorsArray[i * 3 + q];
                    eigenvectorsArray[i * 3 + p] = temp;
                }

            } while (max > Eps);

            OrderByEigenvalues(eigenvectorsArray, eigenvalues);

            eigenvectors = new Matrix3x3d(eigenvectorsArray);

            return FromDiagonal(new Matrix3x3d(eigenvalues));
        }
        public Matrix3x3d Transpose()
        {
            double[] temp = new double[3 * 3];

            for (int i = 0; i < 3; i++)
            {
                for (int j = 0; j < 3; j++)
                {
                    temp[i * 3 + j] = this[j, i];
                }
            }

            return new Matrix3x3d(temp);
        }
        #endregion

        #region arithmetic
        public static Matrix3x3d Multiply(in Matrix3x3d left, in Matrix3x3d right)
        {
            double[] res = new double[3 * 3];

            res[0] = left.M00 * right.M00 + left.M01 * right.M10 + left.M02 * right.M20;
            res[1] = left.M00 * right.M01 + left.M01 * right.M11 + left.M02 * right.M21;
            res[2] = left.M00 * right.M02 + left.M01 * right.M12 + left.M02 * right.M22;

            res[3] = left.M10 * right.M00 + left.M11 * right.M10 + left.M12 * right.M20;
            res[4] = left.M10 * right.M01 + left.M11 * right.M11 + left.M12 * right.M21;
            res[5] = left.M10 * right.M02 + left.M11 * right.M12 + left.M12 * right.M22;

            res[6] = left.M20 * right.M00 + left.M21 * right.M10 + left.M22 * right.M20;
            res[7] = left.M20 * right.M01 + left.M21 * right.M11 + left.M22 * right.M21;
            res[8] = left.M20 * right.M02 + left.M21 * right.M12 + left.M22 * right.M22;

            return new Matrix3x3d(res);
        }

        public static Vector3d Multiply(in Matrix3x3d left, in Vector3d right)
        {
            double v1 = left.M00 * right[0] + left.M01 * right[1] + left.M02 * right[2];
            double v2 = left.M10 * right[0] + left.M11 * right[1] + left.M12 * right[2];
            double v3 = left.M20 * right[0] + left.M21 * right[1] + left.M22 * right[2];

            return new Vector3d(v1, v2, v3);
        }

        public static Matrix3x3d operator *(in Matrix3x3d left, in Matrix3x3d right)
        {
            return Multiply(left, right);
        }

        public static Vector3d operator *(in Matrix3x3d left, in Vector3d right)
        {
            return Multiply(left, right);
        }

        public static bool operator !=(Matrix3x3d left, Matrix3x3d right)
        {
            return !left.Equals(right);
        }

        public static bool operator ==(Matrix3x3d left, Matrix3x3d right)
        {
            return left.Equals(right);
        }

        public bool Equals(Matrix3x3d other)
        {
            if (!M00.Equals(other.M00))
            {
                return false;
            }

            if (!M01.Equals(other.M01))
            {
                return false;
            }

            if (!M02.Equals(other.M02))
            {
                return false;
            }

            if (!M10.Equals(other.M10))
            {
                return false;
            }

            if (!M11.Equals(other.M11))
            {
                return false;
            }

            if (!M12.Equals(other.M12))
            {
                return false;
            }

            if (!M20.Equals(other.M20))
            {
                return false;
            }

            if (!M21.Equals(other.M21))
            {
                return false;
            }

            if (!M22.Equals(other.M22))
            {
                return false;
            }

            return true;
        }

        public override bool Equals(object obj)
        {
            if (obj is Matrix3x3d mat)
            {
                return Equals(mat);
            }

            return false;
        }

        #endregion

        #region util
        public IEnumerator<double> GetEnumerator()
        {
            yield return M00;
            yield return M01;
            yield return M02;
            yield return M10;
            yield return M11;
            yield return M12;
            yield return M20;
            yield return M21;
            yield return M22;
        }

        IEnumerator IEnumerable.GetEnumerator()
        {
            return GetEnumerator();
        }

        public override int GetHashCode()
        {
            return M00.GetHashCode() ^ M01.GetHashCode() ^ M02.GetHashCode()
                 ^ M10.GetHashCode() ^ M11.GetHashCode() ^ M12.GetHashCode()
                 ^ M20.GetHashCode() ^ M21.GetHashCode() ^ M22.GetHashCode();
        }

        public string ToString(string format)
        {
            return "3×3 Matrix:\n"
                + string.Format(CultureInfo.CurrentCulture, format, M00) + "\t" + string.Format(CultureInfo.CurrentCulture, format, M01) + "\t" + string.Format(CultureInfo.CurrentCulture, format, M02) + "\n"
                + string.Format(CultureInfo.CurrentCulture, format, M10) + "\t" + string.Format(CultureInfo.CurrentCulture, format, M11) + "\t" + string.Format(CultureInfo.CurrentCulture, format, M12) + "\n"
                + string.Format(CultureInfo.CurrentCulture, format, M20) + "\t" + string.Format(CultureInfo.CurrentCulture, format, M21) + "\t" + string.Format(CultureInfo.CurrentCulture, format, M22);

        }
        public override string ToString()
        {
            return ToString("{0,-20}");
        }
        #endregion

        #region private methods
        /// <summary>
        /// 行列の非対角成分の絶対値の最大値を与えるインデックスを探す。
        /// Jacobi法で使う。
        /// </summary>
        /// <param name="matrix">行列</param>
        /// <param name="p">行インデックス</param>
        /// <param name="q">列インデックス</param>
        /// <returns>最大値</returns>
        private static double GetMaxValue(double[] matrix, out int p, out int q)
        {
            if (matrix == null)
            {
                throw new ArgumentNullException(nameof(matrix));
            }

            double max = Math.Abs(matrix[0 * 3 + 1]);
            p = 0;
            q = 1;

            for (int i = 0; i < 3; i++)
            {
                for (int j = i + 1; j < 3; j++)
                {
                    double temp = Math.Abs(matrix[i * 3 + j]);
                    if (temp < max)
                    {
                        continue;
                    }

                    max = temp;
                    p = i;
                    q = j;
                }
            }

            return max;
        }

        private static void OrderByEigenvalues(IList<double> eigenvectors, IList<double> eigenvalues)
        {
            for (int i = 0; i < 3; i++)
            {
                for (int j = i; j + 1 < 3; j++)
                {
                    if (eigenvalues[i * 3 + i] < eigenvalues[(j + 1) * 3 + (j + 1)])
                    {
                        ExchangeColumn(eigenvectors, i, j + 1);
                        double tempt = eigenvalues[(j + 1) * 3 + (j + 1)];
                        eigenvalues[(j + 1) * 3 + (j + 1)] = eigenvalues[i * 3 + i];
                        eigenvalues[i * 3 + i] = tempt;
                    }
                }
            }
        }

        private static void ExchangeColumn(IList<double> mat, int i, int j)
        {
            for (int row = 0; row < 3; row++)
            {
                double temp = mat[row * 3 + i];
                mat[row * 3 + i] = mat[row * 3 + j];
                mat[row * 3 + j] = temp;
            }
        }
        #endregion
    }

    public struct SvdMatrices : IEquatable<SvdMatrices>
    {
        internal SvdMatrices(Matrix3x3d u, Matrix3x3d s, Matrix3x3d v)
        {
            U = u;
            S = s;
            V = v;
        }

        public Matrix3x3d S { get; }
        public Matrix3x3d U { get; }
        public Matrix3x3d V { get; }

        public static bool operator !=(SvdMatrices left, SvdMatrices right)
        {
            return !left.Equals(right);
        }

        public static bool operator ==(SvdMatrices left, SvdMatrices right)
        {
            return left.Equals(right);
        }

        public bool Equals(SvdMatrices other)
        {
            return S == other.S && V == other.V && U == other.U;
        }

        public override bool Equals(object obj)
        {
            if (obj is SvdMatrices other)
            {
                return Equals(other);
            }

            return false;
        }
        public override int GetHashCode()
        {
            return S.GetHashCode() ^ V.GetHashCode() ^ U.GetHashCode();
        }
    }
}
