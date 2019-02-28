/*
*
*  Created by Shun Suzuki on 24/07/2018.
*
*/

#if UNITY_2018_3_OR_NEWER
#define UNITY
#endif

using System;
using System.Collections;
using System.Collections.Generic;
using System.Globalization;

#if UNITY
using UnityEngine;
using Vector3f = UnityEngine.Vector3;
#endif

namespace AUTD3Sharp
{
    unsafe public struct Matrix3x3f : IEquatable<Matrix3x3f>, IEnumerable<float>
    {
        #region const
        private const float EPS = 1e-5f;
        #endregion

        #region field
        private float _m00;
        private float _m01;
        private float _m02;
        private float _m10;
        private float _m11;
        private float _m12;
        private float _m20;
        private float _m21;
        private float _m22;
        #endregion

        #region ctor
        public Matrix3x3f(params float[] mat)
        {
            if (mat == null) throw new ArgumentNullException(nameof(mat));
            if (mat.Length != 9) throw new InvalidCastException();

            _m00 = mat[0];
            _m01 = mat[1];
            _m02 = mat[2];
            _m10 = mat[3];
            _m11 = mat[4];
            _m12 = mat[5];
            _m20 = mat[6];
            _m21 = mat[7];
            _m22 = mat[8];
        }

        public Matrix3x3f(Vector3f[] pcX, Vector3f[] pcY)
        {
            if (pcX == null) throw new ArgumentNullException(nameof(pcX));
            if (pcY == null) throw new ArgumentNullException(nameof(pcY));

            if (pcX.Length != pcY.Length) throw new ArgumentException("Size of point sets must be the same.");

            var mat = new float[9];

            fixed (Vector3f* xvp = pcX)
            fixed (Vector3f* yvp = pcY)
            {
                {
                    float* xp = (float*)xvp;
                    float* yp = (float*)yvp;

                    for (int i = 0; i < 3; i++)
                        for (int j = 0; j < 3; j++)
                        {
                            var temp = 0.0f;
                            for (int k = 0; k < pcX.Length; k++)
                                temp += xp[k * 3 + i] * yp[k * 3 + j];
                            mat[i * 3 + j] = temp;
                        }
                }
            }

            _m00 = mat[0];
            _m01 = mat[1];
            _m02 = mat[2];
            _m10 = mat[3];
            _m11 = mat[4];
            _m12 = mat[5];
            _m20 = mat[6];
            _m21 = mat[7];
            _m22 = mat[8];
        }

        public Matrix3x3f(Vector3f[] pcX, Vector3f[] pcY, float[] weight)
        {
            if (pcX == null) throw new ArgumentNullException(nameof(pcX));
            if (pcY == null) throw new ArgumentNullException(nameof(pcY));
            if (weight == null) throw new ArgumentNullException(nameof(weight));

            if (pcX.Length != pcY.Length || pcX.Length != weight.Length) throw new ArgumentException("Size of point sets and weights must be the same.");

            var mat = new float[9];

            fixed (Vector3f* xvp = pcX)
            fixed (Vector3f* yvp = pcY)
            fixed (float* wp = weight)
            {
                float* xp = (float*)xvp;
                float* yp = (float*)yvp;

                for (int i = 0; i < 3; i++)
                    for (int j = 0; j < 3; j++)
                    {
                        var temp = 0.0f;
                        for (int k = 0; k < pcX.Length; k++)
                            temp += xp[k * 3 + i] * yp[k * 3 + j] * wp[k];
                        mat[i * 3 + j] = temp;
                    }
            }

            _m00 = mat[0];
            _m01 = mat[1];
            _m02 = mat[2];
            _m10 = mat[3];
            _m11 = mat[4];
            _m12 = mat[5];
            _m20 = mat[6];
            _m21 = mat[7];
            _m22 = mat[8];
        }
        #endregion

        #region property
        public static Matrix3x3f Identity => CreateDiagonal(1.0f, 1.0f, 1.0f);

        public float Determinant
        {
            get
            {
                var det = 0.0f;

                det += _m00 * _m11 * _m22 + _m02 * _m10 * _m21 + _m01 * _m12 * _m20;
                det -= _m02 * _m11 * _m20 + _m01 * _m10 * _m22 + _m00 * _m12 * _m21;

                return det;
            }
        }
        #endregion

        #region indexcer
        public float this[int row, int col]
        {
            set
            {
                switch (row * 3 + col)
                {
                    case 0: _m00 = value; return;
                    case 1: _m01 = value; return;
                    case 2: _m02 = value; return;

                    case 3: _m10 = value; return;
                    case 4: _m11 = value; return;
                    case 5: _m12 = value; return;

                    case 6: _m20 = value; return;
                    case 7: _m21 = value; return;
                    case 8: _m22 = value; return;

                    default: if (2 < row) throw new ArgumentOutOfRangeException(nameof(row)); else throw new ArgumentOutOfRangeException(nameof(col));
                }
            }
            get
            {
                switch (row * 3 + col)
                {
                    case 0: return _m00;
                    case 1: return _m01;
                    case 2: return _m02;

                    case 3: return _m10;
                    case 4: return _m11;
                    case 5: return _m12;

                    case 6: return _m20;
                    case 7: return _m21;
                    case 8: return _m22;

                    default: if (2 < row) throw new ArgumentOutOfRangeException(nameof(row)); else throw new ArgumentOutOfRangeException(nameof(col));
                }
            }
        }
        #endregion

        #region factory
        public static Matrix3x3f CreateDiagonal(params float[] diag)
        {
            if (diag == null) throw new ArgumentNullException(nameof(diag));
            if (diag.Length != 3) throw new InvalidCastException();

            var mat = new Matrix3x3f
            {
                _m00 = diag[0],
                _m11 = diag[1],
                _m22 = diag[2]
            };
            return mat;
        }
        public static Matrix3x3f CreateDiagonal(Vector3f diag) => CreateDiagonal(diag[0], diag[1], diag[2]);

        public static Vector3f FromDiagonal(Matrix3x3f mat) => new Vector3f(mat[0, 0], mat[1, 1], mat[2, 2]);

        public static Matrix3x3f FromEulerXYZ(float eulerX, float eulerY, float eulerZ)
        {
            var xRot = Matrix3x3f.Identity;
            xRot._m11 = xRot._m22 = (float)Math.Cos(eulerX);
            xRot._m12 = -(float)Math.Sin(eulerX);
            xRot._m21 = (float)Math.Sin(eulerX);

            var yRot = Matrix3x3f.Identity;
            yRot._m00 = yRot._m22 = (float)Math.Cos(eulerY);
            yRot._m02 = (float)Math.Sin(eulerY);
            yRot._m20 = -(float)Math.Sin(eulerY);

            var zRot = Matrix3x3f.Identity;
            zRot._m00 = zRot._m11 = (float)Math.Cos(eulerZ);
            zRot._m01 = -(float)Math.Sin(eulerZ);
            zRot._m10 = (float)Math.Sin(eulerZ);

            return zRot * yRot * xRot;
        }
        #endregion

        #region public methods
        /// <summary>
        /// 特異値計算
        /// </summary>
        /// <returns>結果を格納した構造体</returns>
        public SVDMatrices Svd()
        {
            var X = this;

            var XtX = X.Transpose() * X;
            var eigenNum1 = Jacobi(XtX, out Matrix3x3f V);

            var XXt = X * X.Transpose();
            var eigenNum2 = Jacobi(XXt, out Matrix3x3f U);

            var eigenNum = ((eigenNum1 + eigenNum2) / 2);
            eigenNum.Rectify();

            var S = CreateDiagonal((float)Math.Sqrt(eigenNum[0]), (float)Math.Sqrt(eigenNum[1]), (float)Math.Sqrt(eigenNum[2]));

            var Sp = U.Transpose() * X * V;

            var H = new Matrix3x3f
            {
                _m00 = Sp._m00 < 0 ? -1 : 1,
                _m11 = Sp._m11 < 0 ? -1 : 1,
                _m22 = Sp._m22 < 0 ? -1 : 1
            };
            V = V * H;

            return new SVDMatrices(U, S, V);
        }

        /// <summary>
        /// Jacobi法による固有値および固有ベクトルの計算。
        /// 対称行列のみ適用可。
        /// </summary>
        /// <param name="symmetricMatrix">対称行列</param>
        /// <param name="eigenVectors">固有ベクトルの行列</param>
        /// <returns></returns>
        public static Vector3f Jacobi(Matrix3x3f symmetricMatrix, out Matrix3x3f eigenVectors)
        {
            float max;
            float app, apq, aqq;
            float alpha, beta, gamma;
            float s, c;

            eigenVectors = CreateDiagonal(1.0f, 1.0f, 1.0f);

            var count = 0;
            var eigenNums = symmetricMatrix;

            do
            {
                count++;
                if (count > 1000) break;

                if ((max = GetMaxValue(ref eigenNums, out int p, out int q)) == 0) break;

                app = eigenNums[p, p];
                apq = eigenNums[p, q];
                aqq = eigenNums[q, q];

                alpha = (app - aqq) / 2;
                beta = -apq;
                gamma = (float)(Math.Abs(alpha) / Math.Sqrt(alpha * alpha + beta * beta));

                s = (float)Math.Sqrt((1 - gamma) / 2);
                c = (float)Math.Sqrt((1 + gamma) / 2);
                if (alpha * beta < 0) s = -s;

                for (var i = 0; i < 3; i++)
                {
                    var temp = c * eigenNums[p, i] - s * eigenNums[q, i];
                    eigenNums[q, i] = s * eigenNums[p, i] + c * eigenNums[q, i];
                    eigenNums[p, i] = temp;
                }

                for (var i = 0; i < 3; i++)
                {
                    eigenNums[i, p] = eigenNums[p, i];
                    eigenNums[i, q] = eigenNums[q, i];
                }

                eigenNums[p, p] = c * c * app + s * s * aqq - 2 * s * c * apq;
                eigenNums[p, q] = s * c * (app - aqq) + (c * c - s * s) * apq;
                eigenNums[q, p] = s * c * (app - aqq) + (c * c - s * s) * apq;
                eigenNums[q, q] = s * s * app + c * c * aqq + 2 * s * c * apq;

                for (var i = 0; i < 3; i++)
                {
                    var temp = c * eigenVectors[i, p] - s * eigenVectors[i, q];
                    eigenVectors[i, q] = s * eigenVectors[i, p] + c * eigenVectors[i, q];
                    eigenVectors[i, p] = temp;
                }

            } while (max > EPS);

            OrderByEigenNums(ref eigenVectors, ref eigenNums);

            return FromDiagonal(eigenNums);
        }
        public Matrix3x3f Transpose()
        {
            var floatempMat = new float[3 * 3];

            for (int i = 0; i < 3; i++)
                for (int j = 0; j < 3; j++)
                    floatempMat[i * 3 + j] = this[j, i];
            return new Matrix3x3f(floatempMat);
        }
        #endregion

        #region arithmetic
        public static Matrix3x3f Multiply(in Matrix3x3f left, in Matrix3x3f right)
        {
            var res = new float[3 * 3];

            res[0] = left._m00 * right._m00 + left._m01 * right._m10 + left._m02 * right._m20;
            res[1] = left._m00 * right._m01 + left._m01 * right._m11 + left._m02 * right._m21;
            res[2] = left._m00 * right._m02 + left._m01 * right._m12 + left._m02 * right._m22;

            res[3] = left._m10 * right._m00 + left._m11 * right._m10 + left._m12 * right._m20;
            res[4] = left._m10 * right._m01 + left._m11 * right._m11 + left._m12 * right._m21;
            res[5] = left._m10 * right._m02 + left._m11 * right._m12 + left._m12 * right._m22;

            res[6] = left._m20 * right._m00 + left._m21 * right._m10 + left._m22 * right._m20;
            res[7] = left._m20 * right._m01 + left._m21 * right._m11 + left._m22 * right._m21;
            res[8] = left._m20 * right._m02 + left._m21 * right._m12 + left._m22 * right._m22;

            return new Matrix3x3f(res);
        }

        public static Vector3f Multiply(in Matrix3x3f left, in Vector3f right)
        {
            var v1 = left._m00 * right[0] + left._m01 * right[1] + left._m02 * right[2];
            var v2 = left._m10 * right[0] + left._m11 * right[1] + left._m12 * right[2];
            var v3 = left._m20 * right[0] + left._m21 * right[1] + left._m22 * right[2];

            return new Vector3f(v1, v2, v3);
        }

        public static Matrix3x3f operator *(in Matrix3x3f left, in Matrix3x3f right) => Multiply(left, right);
        public static Vector3f operator *(in Matrix3x3f left, in Vector3f right) => Multiply(left, right);

        public static bool operator !=(Matrix3x3f left, Matrix3x3f right) => !left.Equals(right);
        public static bool operator ==(Matrix3x3f left, Matrix3x3f right) => left.Equals(right);

        public bool Equals(Matrix3x3f other)
        {
            if (this._m00 != other._m00) return false;
            if (this._m01 != other._m01) return false;
            if (this._m02 != other._m02) return false;
            if (this._m10 != other._m10) return false;
            if (this._m11 != other._m11) return false;
            if (this._m12 != other._m12) return false;
            if (this._m20 != other._m20) return false;
            if (this._m21 != other._m21) return false;
            if (this._m22 != other._m22) return false;
            return true;
        }

        public override bool Equals(object obj)
        {
            if (obj is Matrix3x3f mat)
                return this.Equals(mat);
            return false;
        }

        #endregion

        #region util
        public IEnumerator<float> GetEnumerator()
        {
            yield return _m00;
            yield return _m01;
            yield return _m02;
            yield return _m10;
            yield return _m11;
            yield return _m12;
            yield return _m20;
            yield return _m21;
            yield return _m22;
        }

        IEnumerator IEnumerable.GetEnumerator()
        {
            return this.GetEnumerator();
        }

        public override int GetHashCode()
        {
            var hash = 0;
            hash = this._m00.GetHashCode() ^ this._m01.GetHashCode() ^ this._m02.GetHashCode()
                ^ this._m10.GetHashCode() ^ this._m11.GetHashCode() ^ this._m12.GetHashCode()
                ^ this._m20.GetHashCode() ^ this._m21.GetHashCode() ^ this._m22.GetHashCode();
            return hash;
        }

        public string ToString(string format)
        {
            return "3×3 Matrix:\n"
                + String.Format(CultureInfo.CurrentCulture, format, _m00) + "\t" + String.Format(CultureInfo.CurrentCulture, format, _m01) + "\t" + String.Format(CultureInfo.CurrentCulture, format, _m02) + "\n"
                + String.Format(CultureInfo.CurrentCulture, format, _m10) + "\t" + String.Format(CultureInfo.CurrentCulture, format, _m11) + "\t" + String.Format(CultureInfo.CurrentCulture, format, _m12) + "\n"
                + String.Format(CultureInfo.CurrentCulture, format, _m20) + "\t" + String.Format(CultureInfo.CurrentCulture, format, _m21) + "\t" + String.Format(CultureInfo.CurrentCulture, format, _m22);

        }
        public override string ToString() => ToString("{0,-20}");
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
        private static float GetMaxValue(ref Matrix3x3f matrix, out int p, out int q)
        {
            int i, j;
            float max;
            float temp;

            max = Math.Abs(matrix[0, 1]);
            p = 0;
            q = 1;

            for (i = 0; i < 3; i++)
                for (j = i + 1; j < 3; j++)
                {
                    temp = Math.Abs(matrix[i, j]);

                    if (temp > max)
                    {
                        max = temp;
                        p = i;
                        q = j;
                    }
                }
            return max;
        }

        private static void OrderByEigenNums(ref Matrix3x3f eigenVectors, ref Matrix3x3f eigenNums)
        {

            for (int i = 0; i < 3; i++)
                for (int j = i; j + 1 < 3; j++)
                    if (eigenNums[i, i] < eigenNums[j + 1, j + 1])
                    {
                        eigenVectors.ExchangeColumn(i, j + 1);
                        var tempt = eigenNums[j + 1, j + 1];
                        eigenNums[j + 1, j + 1] = eigenNums[i, i];
                        eigenNums[i, i] = tempt;
                    }
        }

        private void ExchangeColumn(int i, int j)
        {
            for (var row = 0; row < 3; row++)
            {
                var temp = this[row, i];
                this[row, i] = this[row, j];
                this[row, j] = temp;
            }
        }
        #endregion
    }

    public struct SVDMatrices
    {
        internal SVDMatrices(Matrix3x3f U, Matrix3x3f S, Matrix3x3f V)
        {
            this.U = U;
            this.S = S;
            this.V = V;
        }

        public Matrix3x3f S { get; }
        public Matrix3x3f U { get; }
        public Matrix3x3f V { get; }

        public static bool operator !=(SVDMatrices left, SVDMatrices right) => !left.Equals(right);
        public static bool operator ==(SVDMatrices left, SVDMatrices right) => left.Equals(right);

        public bool Equals(SVDMatrices other) => this.S == other.S && this.V == other.V && this.U == other.U;

        public override bool Equals(object obj)
        {
            if (obj is SVDMatrices svdmatrices)
                return this.Equals(svdmatrices);
            return false;
        }
        public override int GetHashCode() => this.S.GetHashCode() ^ this.V.GetHashCode() ^ this.U.GetHashCode();
    }
}
