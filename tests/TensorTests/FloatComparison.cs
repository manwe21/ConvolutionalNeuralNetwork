using System;

namespace CpuTensorTests
{
    public static class FloatComparison
    {
        private const float Tolerance = 1e-4f;

        public static bool AreEqual(float a, float b)
        {
            return Math.Abs(a - b) < Tolerance;
        }

        public static bool AreEqual(float[] a, float[] b)
        {
            if (a == null || b == null)
                return false;
            
            if (a.Length != b.Length)
                return false;
            
            for (int i = 0; i < a.Length; i++)
            {
                if (!AreEqual(a[i], b[i]))
                    return false;
            }

            return true;
        }
    }
}
