/*
*
*  Created by Shun Suzuki on 02/07/2018.
*  Copyright © 2018 Hapis Lab. All rights reserved.
*
*/

#if UNITY_2018_3_OR_NEWER
#define UNITY
#endif

#if UNITY
#define LEFT_HANDED
#define DIMENSION_M
#else
#define RIGHT_HANDED
#define DIMENSION_MM
#endif

using Microsoft.Win32.SafeHandles;
using System;
using System.Collections.Generic;
using System.Reflection;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;

#if UNITY
using UnityEngine;
using Vector3f = UnityEngine.Vector3;
using Quaternionf = UnityEngine.Quaternion;
#endif

[assembly: CLSCompliant(false), ComVisible(false)]
[assembly: AssemblyVersion("3.2.1.0")]
namespace AUTD3Sharp
{
    [ComVisible(false)]
    public class Gain : SafeHandleZeroOrMinusOneIsInvalid
    {
        internal IntPtr GainPtr => handle;

        public Gain(IntPtr gain) : base(true)
        {
            SetHandle(gain);
        }

        protected override bool ReleaseHandle()
        {
            NativeMethods.AUTDDeleteGain(handle);
            return true;
        }

        public void Fix()
        {
            NativeMethods.AUTDFixGain(handle);
        }
    }

    [ComVisible(false)]
    public class Modulation : SafeHandleZeroOrMinusOneIsInvalid
    {
        public Modulation(IntPtr modulation) : base(true)
        {
            SetHandle(modulation);
        }
        protected override bool ReleaseHandle()
        {
            NativeMethods.AUTDDeleteModulation(handle);
            return true;
        }
    }

    internal class AUTDControllerHandle : SafeHandleZeroOrMinusOneIsInvalid
    {
        public AUTDControllerHandle(bool ownsHandle) : base(ownsHandle)
        {
            handle = new IntPtr();
            NativeMethods.AUTDCreateController(out handle);
        }

        protected override bool ReleaseHandle()
        {
            NativeMethods.AUTDFreeController(handle);
            return true;
        }
    }

    public sealed class AUTD : IDisposable
    {
        #region const
        public static readonly float UltrasoundWavelength = 8.5f;

        public const int NumTransIndevice = 249;
        public const float AUTDWidth = 192.0f;
        public const float AUTDHeight = 151.4f;

        public const float Pi = 3.14159265f;

#if UNITY
        public readonly static float MeterScale = 1000f;
#endif
        #endregion

        #region field
        private bool _isDisposed = false;
        private AUTDControllerHandle _autdControllerHandle;

        // そのうちなんとかしたい
        // 現状はNative/マイコン側の実装の問題
        private List<Gain> _lateralGains = null;

        #endregion

        #region Controller
        public AUTD()
        {
            _autdControllerHandle = new AUTDControllerHandle(true);
        }

        public int Open() => Open("");
        public int Open(string location) => NativeMethods.AUTDOpenController(_autdControllerHandle, location);

        public int AddDevice(float x, float y, float z, float rz1, float ry, float rz2) => AddDevice(new Vector3f(x, y, z), new Vector3f(rz1, ry, rz2), 0);
        public int AddDevice(float x, float y, float z, float rz1, float ry, float rz2, int groupID) => AddDevice(new Vector3f(x, y, z), new Vector3f(rz1, ry, rz2), groupID);
        public int AddDevice(Vector3f position, Vector3f rotation) => AddDevice(position, rotation, 0);
        public int AddDevice(Vector3f position, Vector3f rotation, int groupID)
        {
            AdjustVector(ref position);
            var res = NativeMethods.AUTDAddDevice(_autdControllerHandle, position[0], position[1], position[2], rotation[0], rotation[1], rotation[2], groupID);
            return res;
        }
        public void DelDevice(int devId)
        {
            NativeMethods.AUTDDelDevice(_autdControllerHandle, devId);
        }
        public void Close()
        {
            NativeMethods.AUTDCloseController(_autdControllerHandle);
        }

        public void Dispose()
        {
            Dispose(true);
            GC.SuppressFinalize(this);
        }

        private void Dispose(bool dispoing)
        {
            if (_isDisposed) return;

            if (dispoing)
            {
                Close();
            }

            _autdControllerHandle.Dispose();

            _isDisposed = true;
        }

        public void SetSilentMode(bool mode)
        {
            NativeMethods.AUTDSetSilentMode(_autdControllerHandle, mode);
        }

        ~AUTD()
        {
            Dispose(false);
        }
        #endregion

        #region Property
        public bool IsOpen
        {
            get
            {
                var res = NativeMethods.AUTDIsOpen(_autdControllerHandle);
                return res;
            }
        }
        public bool IsSilentMode
        {
            get
            {
                var res = NativeMethods.AUTDIsSilentMode(_autdControllerHandle);
                return res;
            }
        }
        public int NumDevices
        {
            get
            {
                var res = NativeMethods.AUTDNumDevices(_autdControllerHandle);
                return res;
            }
        }
        public int NumTrnsducers
        {
            get
            {
                var res = NativeMethods.AUTDNumTransducers(_autdControllerHandle);
                return res;
            }
        }
        public float Frequency
        {
            get
            {
                var res = NativeMethods.AUTDFreqency(_autdControllerHandle);
                return res;
            }
        }
        public long RemainingInBuffer
        {
            get
            {
                var res = NativeMethods.AUTDRemainingInBuffer(_autdControllerHandle);
                return res;
            }
        }
        #endregion

        #region Gain
        public static Gain FocalPointGain(float posX, float posY, float posZ, byte amp)
        {
            NativeMethods.AUTDFocalPointGain(out IntPtr gainPtr, posX, posY, posZ, amp);
            return new Gain(gainPtr);
        }
        public static Gain FocalPointGain(float posX, float posY, float posZ) => FocalPointGain(posX, posY, posZ, 255);
        public static Gain FocalPointGain(Vector3f point, byte amp) => FocalPointGain(point[0], point[1], point[2], amp);
        public static Gain FocalPointGain(Vector3f point)
        {
            AdjustVector(ref point);
            NativeMethods.AUTDFocalPointGain(out IntPtr gainPtr, point[0], point[1], point[2], 255);
            return new Gain(gainPtr);
        }

        public static unsafe Gain GroupedGain(GainMap gainMap)
        {
            if (gainMap == null) throw new ArgumentNullException(nameof(gainMap));

            IntPtr* gainptr = gainMap.GainPointer;
            int* idptr = gainMap.IdPointer;
            NativeMethods.AUTDGroupedGain(out IntPtr gainPtr, idptr, gainptr, gainMap.Size);
            return new Gain(gainPtr);
        }
        public static Gain GroupedGain(params GainPair[] gainPairs) => GroupedGain(new GainMap(gainPairs));

        public static Gain BesselBeamGain(float startPosX, float startPosY, float startPosZ, float dirX, float dirY, float dirZ, float thetaZ)
            => BesselBeamGain(new Vector3f(startPosX, startPosY, startPosZ), new Vector3f(dirX, dirY, dirZ), thetaZ);
        public static Gain BesselBeamGain(Vector3f point, Vector3f dir, float thetaZ)
        {
            AdjustVector(ref point);
            AdjustVector(ref dir);

            NativeMethods.AUTDBesselBeamGain(out IntPtr gainPtr, point[0], point[1], point[2], dir[0], dir[1], dir[2], thetaZ);
            return new Gain(gainPtr);
        }
        public static Gain PlaneWaveGain(float dirX, float dirY, float dirZ)
            => PlaneWaveGain(new Vector3f(dirX, dirY, dirZ));
        public static Gain PlaneWaveGain(Vector3f dir)
        {
            AdjustVector(ref dir);

            NativeMethods.AUTDPlaneWaveGain(out IntPtr gainPtr, dir[0], dir[1], dir[2]);
            return new Gain(gainPtr);
        }
        public static Gain MatlabGain(string fileName, string varname)
        {
            NativeMethods.AUTDMatlabGain(out IntPtr gainPtr, fileName, varname);
            return new Gain(gainPtr);
        }
        unsafe static public Gain HoloGain(Vector3f[] focuses, float[] amps)
        {
            if (focuses == null) throw new ArgumentNullException(nameof(focuses));
            if (amps == null) throw new ArgumentNullException(nameof(amps));

            var size = amps.Length;
            var foci = new float[size * 3];
            for (int i = 0; i < size; i++)
            {
                AdjustVector(ref focuses[i]);

                foci[3 * i] = focuses[i][0];
                foci[3 * i + 1] = focuses[i][1];
                foci[3 * i + 2] = focuses[i][2];
            }

            var gainPtr = new IntPtr();

            fixed (float* fp = &foci[0])
            fixed (float* ap = &amps[0])
            {
                NativeMethods.AUTDHoloGain(out gainPtr, fp, ap, size);
            }
            return new Gain(gainPtr);
        }
        public static Gain NullGain()
        {
            NativeMethods.AUTDNullGain(out IntPtr gainPtr);
            return new Gain(gainPtr);
        }

        [System.Diagnostics.CodeAnalysis.SuppressMessage("Microsoft.Performance", "CA1814:PreferJaggedArraysOverMultidimensional", MessageId = "0#")]
        unsafe public Gain CustomGain(ushort[,] data)
        {
            if (data == null) throw new ArgumentNullException(nameof(data));

            var numDev = NumDevices;

            if (data.GetLength(0) != numDev) throw new ArgumentOutOfRangeException("Invalid data length. " + numDev + " AUTDs was added.");
            if (data.GetLength(1) != NumTransIndevice) throw new ArgumentOutOfRangeException("Some Device have wrong Data length. A device must have " + NumTransIndevice + " data.");

            var gainPtr = new IntPtr();
            var length = data.GetLength(0) * data.GetLength(1);
            fixed (ushort* r = data) NativeMethods.AUTDCustomGain(out gainPtr, r, length);

            return new Gain(gainPtr);
        }
        #endregion

        #region Modulation
        public static Modulation Modulation() => Modulation(255);
        public static Modulation Modulation(byte amp)
        {
            NativeMethods.AUTDModulation(out IntPtr modPtr, amp);
            return new Modulation(modPtr);
        }
        public static Modulation RawPCMModulation(string fileName) => RawPCMModulation(fileName, 0.0f);
        public static Modulation RawPCMModulation(string fileName, float samplingFreq)
        {
            NativeMethods.AUTDRawPCMModulation(out IntPtr modPtr, fileName, samplingFreq);
            return new Modulation(modPtr);
        }
        public static Modulation SawModulation(float freq)
        {
            NativeMethods.AUTDSawModulation(out IntPtr modPtr, freq);
            return new Modulation(modPtr);
        }
        public static Modulation SineModulation(float freq) => SineModulation(freq, 1, 0.5f);
        public static Modulation SineModulation(float freq, float amp) => SineModulation(freq, amp, 0.5f);
        public static Modulation SineModulation(float freq, float amp, float offset)
        {
            NativeMethods.AUTDSineModulation(out IntPtr modPtr, freq, amp, offset);
            return new Modulation(modPtr);
        }
        #endregion

        #region LowLevelInterface
        public void AppendGain(Gain gain)
        {
            if (gain == null) throw new ArgumentNullException(nameof(gain));

            NativeMethods.AUTDAppendGain(_autdControllerHandle, gain);
        }
        public void AppendGainSync(Gain gain)
        {
            if (gain == null) throw new ArgumentNullException(nameof(gain));

            NativeMethods.AUTDAppendGainSync(_autdControllerHandle, gain);
        }
        public void AppendModulation(Modulation mod)
        {
            if (mod == null) throw new ArgumentNullException(nameof(mod));

            NativeMethods.AUTDAppendModulation(_autdControllerHandle, mod);
        }
        public void AppendModulationSync(Modulation mod)
        {
            if (mod == null) throw new ArgumentNullException(nameof(mod));

            NativeMethods.AUTDAppendModulationSync(_autdControllerHandle, mod);
        }
        public void AppendLateralGain(Gain gain)
        {
            if (gain == null) throw new ArgumentNullException(nameof(gain));

            if (_lateralGains == null) _lateralGains = new List<Gain>();
            _lateralGains.Add(gain);

            NativeMethods.AUTDAppendLateralGain(_autdControllerHandle, gain);
        }
        public void AppendLateralGain(params Gain[] gainList)
        {
            if (gainList == null) throw new ArgumentNullException(nameof(gainList));

            foreach (var gain in gainList) AppendLateralGain(gain);
        }
        public void StartLateralModulation(float freq)
        {
            NativeMethods.AUTDStartLateralModulation(_autdControllerHandle, freq);
        }
        public void FinishLateralModulation()
        {
            NativeMethods.AUTDFinishLateralModulation(_autdControllerHandle);
        }
        public void ResetLateralGain()
        {
            if (_lateralGains != null) _lateralGains.Clear();
            NativeMethods.AUTDResetLateralGain(_autdControllerHandle);
        }
        public void SetGain(int deviceIndex, int transIndex, int amp, int phase)
        {
            NativeMethods.AUTDSetGain(_autdControllerHandle, deviceIndex, transIndex, amp, phase);
        }
        public void Flush()
        {
            NativeMethods.AUTDFlush(_autdControllerHandle);
        }
        public int DeviceIdForDeviceIndex(int devIdx)
        {
            var res = NativeMethods.AUTDDevIdForDeviceIdx(_autdControllerHandle, devIdx);
            return res;
        }
        public int DeviceIdForDTransducerIndex(int transIdx)
        {
            var res = NativeMethods.AUTDDevIdForTransIdx(_autdControllerHandle, transIdx);
            return res;
        }
        unsafe public Vector3f TransPostion(int transIdx)
        {
            var fp = NativeMethods.AUTDTransPosition(_autdControllerHandle, transIdx);
            return new Vector3f(fp[0], fp[1], fp[2]);
        }
        unsafe public Vector3f TransDirection(int transIdx)
        {
            var fp = NativeMethods.AUTDTransDirection(_autdControllerHandle, transIdx);
            return new Vector3f(fp[0], fp[1], fp[2]);
        }
        unsafe public static double[] GetEulerAngleZYZ(double[] rot)
        {
            var angs = new double[3];

            fixed (double* r = rot)
            {
                var ang = NativeMethods.GetAngleZYZ(r);

                angs[0] = ang[0];
                angs[1] = ang[1];
                angs[2] = ang[2];
            }
            return angs;
        }
        #endregion

        #region DEBUG
#if DEBUG
        public static void SetDebugLogFunc(Action<string> debugLogFunc)
        {
            var callback = new NativeMethods.DebugLogDelegate(debugLogFunc);
            var funcPtr = Marshal.GetFunctionPointerForDelegate(callback);
            NativeMethods.SetDebugLog(funcPtr);
            NativeMethods.DebugLogTest();
        }

        public static void DebugLog(string msg)
        {
            NativeMethods.DebugLog(msg);
        }
#endif
        #endregion

        #region GeometryAdjust
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static void AdjustVector(ref Vector3f vector)
        {
#if LEFT_HANDED
            vector[0] = -vector[0];
#endif
#if DIMENSION_M
            vector[0] *= MeterScale;
            vector[1] *= MeterScale;
            vector[2] *= MeterScale;
#endif
        }
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static void AdjustQuaternion(ref Quaternionf quaternion)
        {
#if LEFT_HANDED
            quaternion[0] = -quaternion[0];
            quaternion.w = -quaternion.w;
#endif
        }
        #endregion
    }
}
