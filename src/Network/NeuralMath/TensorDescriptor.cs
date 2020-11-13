using System.Runtime.InteropServices;

namespace Network.NeuralMath
{
    [StructLayout(LayoutKind.Sequential)]
    public struct TensorDescriptor
    {
        public int Batch;
        public int Channels;
        public int Height;
        public int Width;
        public int Size;

        public TensorDescriptor(int batch, int channels, int height, int width, int size)
        {
            Batch = batch;
            Channels = channels;
            Height = height;
            Width = width;
            Size = size;
        }

    }
}
