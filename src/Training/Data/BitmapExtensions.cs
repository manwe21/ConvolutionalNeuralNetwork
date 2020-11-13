using System.Drawing;
using System.Drawing.Imaging;

namespace Training.Data
{
    public static class BitmapExtensions
    {
        public static unsafe void ToArray(this Bitmap map, byte[] res)
        {
            BitmapData bitmapData = map.LockBits(new Rectangle(0, 0, map.Width, map.Height), ImageLockMode.ReadWrite, map.PixelFormat);
            int bitsPerPixel = Image.GetPixelFormatSize(map.PixelFormat);

            int mapSize = bitmapData.Height * bitmapData.Width;
            int n = 0;
            byte* scan0 = (byte*)bitmapData.Scan0.ToPointer();
            for (int i = 0; i < bitmapData.Height; ++i)
            {
                for (int j = 0; j < bitmapData.Width; ++j)
                {
                    byte* data = scan0 + i * bitmapData.Stride + j * bitsPerPixel / 8;
                    res[n] = data[2];
                    res[n + mapSize] = data[1];
                    res[n + 2 * mapSize] = data[0];
                    n++;
                }
            }

            map.UnlockBits(bitmapData);
        }
    }
}
