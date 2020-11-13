using System;
using System.Collections.Generic;
using Network.NeuralMath;
using Network.NeuralMath.Gpu;

namespace Training.Optimizers.Gpu
{
    public class GpuGradientDescent : GradientDescent
    {
        public GpuGradientDescent(float learningRate) : base(learningRate)
        { }

        public override void Correct(Tensor weights, Tensor gradients, Dictionary<string, Tensor> parameters,
            bool resetDw, int iteration)
        {
            var wStorage = weights.Storage as GpuStorage;
            var gStorage = gradients.Storage as GpuStorage;
            
            if(wStorage == null || gStorage == null)
                throw new ArgumentException("Unsupported storage");
            
            var context = wStorage.Context;
            
            context.KernelManager.LaunchKernel(
                "gradientDescent",
                weights.Size,
                0,
                wStorage.DeviceStorage.DevicePointer,
                gStorage.DeviceStorage.DevicePointer,
                LearningRate,
                weights.Size);
        }
    }
}
