using System;
using System.Collections.Generic;
using ManagedCuda.NPP;
using Network.NeuralMath;
using Network.NeuralMath.Gpu;

namespace Training.Optimizers.Gpu
{
    public class GpuAdaGrad : AdaGrad
    {
        public GpuAdaGrad(float learningRate) : base(learningRate)
        {
        }

        public override void Correct(Tensor weights, Tensor gradients, Dictionary<string, Tensor> parameters,
            bool resetDw, int iteration)    
        {
            var wStorage = weights.Storage as GpuStorage;
            var gStorage = gradients.Storage as GpuStorage;
            var historyStorage = parameters["GradientHistory"].Storage as GpuStorage;
            
            if(wStorage == null || gStorage == null || historyStorage == null)
                throw new ArgumentException("Unsupported storage");
            
            var context = wStorage.Context;
            
            context.KernelManager.LaunchKernel(
                "adaGrad",
                weights.Size,
                0,
                wStorage.DeviceStorage.DevicePointer,
                gStorage.DeviceStorage.DevicePointer,
                historyStorage.DeviceStorage.DevicePointer,
                LearningRate,
                weights.Size);
        }
    }
}
