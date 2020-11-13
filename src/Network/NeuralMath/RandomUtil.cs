using System;

namespace Network.NeuralMath
{
    public static class RandomUtil
    {
        private static readonly Random Rand = new Random();

        public static double GetRandomNumber(double minValue, double maxValue)
        {
            return minValue + Rand.NextDouble() * (maxValue - minValue);
        }

        public static double GetRandomNumber()
        {
            return Rand.NextDouble();
        }
        
        //Box - Muller transform
        public static double GetGaussian(double mean, double deviation)
        {
            double u1 = 1.0 - Rand.NextDouble();
            double u2 = 1.0 - Rand.NextDouble();
            
            double z = Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Sin(2.0 * Math.PI * u2);
            return mean + deviation * z;
        }
        
        //Box - Muller transform (second variation)
        public static double GetGaussian2(int mean, double deviation)
        {
            double u1;
            double u2;
            double s;
            do
            {
                u1 = -1.0 + Rand.NextDouble() * 2;
                u2 = -1.0 + Rand.NextDouble() * 2;
                s = u1 * u1 + u2 * u2;
            } while (s == 0 || s > 1);
            
            double rand = u1 * Math.Sqrt(-2 * Math.Log(s) / s);
            return mean + deviation * rand;
        }
        
    }
}
