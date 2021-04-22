using System;
using System.Collections.Generic;
using Network.NeuralMath;
using Network.NeuralMath.Gpu;

namespace Training.Optimizers.Gpu
{
    public class GpuAdam : Adam
    {
        public GpuAdam(float learningRate) : base(learningRate)
        { }
        
        public override void Correct(Tensor weights, Tensor gradients, Dictionary<string, Tensor> parameters,
            bool resetDw, int iteration)
        {
            var wStorage = weights.Storage as GpuStorage;
            var gStorage = gradients.Storage as GpuStorage;
            var sStorage = parameters["S"].Storage as GpuStorage;
            var dStorage = parameters["D"].Storage as GpuStorage;
            
            if(wStorage == null || gStorage == null || sStorage == null || dStorage == null)
                throw new ArgumentException("Unsupported storage");
            
            var context = wStorage.Context;
            
            context.KernelManager.LaunchKernel(
                "adam",
                weights.Size,
                0,
                wStorage.DeviceStorage.DevicePointer,
                gStorage.DeviceStorage.DevicePointer,
                sStorage.DeviceStorage.DevicePointer,
                dStorage.DeviceStorage.DevicePointer,
                LearningRate,
                Alpha,
                Beta,
                iteration,
                weights.Size);
        }

    }
}
