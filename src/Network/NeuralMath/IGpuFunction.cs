namespace Network.NeuralMath
{
    public interface IGpuFunction
    {
        string ForwardKernelName { get; }
        string BackwardKernelName { get; }
    }
}
