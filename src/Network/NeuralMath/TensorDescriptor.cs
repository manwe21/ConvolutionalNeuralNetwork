using System.Runtime.InteropServices;

namespace Network.NeuralMath
{
    [StructLayout(LayoutKind.Sequential)]
    public struct TensorDescriptor
    {
        public readonly int Batch;
        public readonly int Channels;
        public readonly int Height;
        public readonly int Width;
        public readonly int Size;

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
