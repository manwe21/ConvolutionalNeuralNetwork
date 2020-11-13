using Network.NeuralMath;
using Network.NeuralMath.Cpu;

namespace Demo
{
    static class Program    
    {
        static void Main(string[] args)
        {
            Tensor t = new CpuTensor(new CpuStorage(new Shape(1, 1, 20, 20)));
            Tensor t2 = new CpuTensor(new CpuStorage(new Shape(1, 1, 20, 20)));
            t.Dot2D(t2, new CpuTensor());
        }

    }
}
