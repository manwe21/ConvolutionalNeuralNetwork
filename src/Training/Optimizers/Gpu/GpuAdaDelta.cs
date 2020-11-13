using System;
using System.Collections.Generic;
using Network.NeuralMath;
using Network.NeuralMath.Gpu;

namespace Training.Optimizers.Gpu
{
    public class GpuAdaDelta : AdaDelta
    {
        public GpuAdaDelta(float learningRate) : base(learningRate)
        {
        }

        public override void Correct(Tensor weights, Tensor gradients, Dictionary<string, Tensor> parameters,
            bool resetDw, int iteration)
        {
            var wStorage = weights.Storage as GpuStorage;
            var gStorage = gradients.Storage as GpuStorage;
            var esqStorage = parameters["Esq"].Storage as GpuStorage;
            
            if(wStorage == null || gStorage == null || esqStorage == null)
                throw new ArgumentException("Unsupported storage");
            
            var context = wStorage.Context;
            
            context.KernelManager.LaunchKernel(
                "adaDelta",
                weights.Size,
                0,
                wStorage.DeviceStorage.DevicePointer,
                gStorage.DeviceStorage.DevicePointer,
                esqStorage.DeviceStorage.DevicePointer,
                LearningRate,
                Gamma,
                weights.Size);
        }
    }
}
