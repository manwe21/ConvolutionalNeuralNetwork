using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using ManagedCuda;
using ManagedCuda.BasicTypes;

namespace Network.NeuralMath.Gpu
{
    public class KernelManager
    {
        private readonly GpuContext _context;
        private readonly Dictionary<string, CudaKernel> _kernels;

        private readonly int _maxThreads;
    
        public KernelManager(GpuContext context)
        {
            _kernels = new Dictionary<string, CudaKernel>();
            _context = context;
            _maxThreads = context.CudaContext.GetDeviceInfo().MaxThreadsPerBlock;
            
            LoadAllKernels();
        }

        private void LoadAllKernels()
        {
            LoadAllKernelsFromModule("activation.ptx");
            LoadAllKernelsFromModule("conv.ptx");
            LoadAllKernelsFromModule("loss.ptx");
            LoadAllKernelsFromModule("basic.ptx");
            LoadAllKernelsFromModule("padding.ptx");
            LoadAllKernelsFromModule("pooling.ptx");
            LoadAllKernelsFromModule("optimizers.ptx");
            LoadAllKernelsFromModule("softmax.ptx");
        }

        public void LoadAllKernelsFromModule(string modulePath) 
        {    
            if(modulePath == null || !File.Exists(modulePath))
                throw new ArgumentException(nameof(modulePath));
            
            List<string> lines = new List<string>();
            string pattern = "entry ";
            
            using var stream = new FileStream(modulePath, FileMode.Open);
            using var reader = new StreamReader(stream);
            while (!reader.EndOfStream)
            {
                string line = reader.ReadLine();
                if(line.Contains(pattern))
                    lines.Add(line);
            }

            List<string> moduleNames = lines.Select(l =>
            {
                int i = l.IndexOf(pattern);
                return l.Substring(i + pattern.Length, l.Length - pattern.Length - i - 1);
            }).ToList();
            
            foreach (var name in moduleNames)
            {
                LoadKernel(modulePath, name);
            }
        }

        private void LoadKernel(string modulePath, string kernelName)
        {
            var kernel = _context.CudaContext.LoadKernel(modulePath, kernelName);
            _kernels.Add(kernelName, kernel);
        }    
        
        public void LoadKernel(CUmodule module, string kernelName)
        {
            var kernel = new CudaKernel(kernelName, module, _context.CudaContext);
            _kernels.Add(kernelName, kernel);
        }
        
        public void LoadKernel(CudaKernel kernel)
        {
            _kernels.Add(kernel.KernelName, kernel);
        }
        
        private void CalcDimensions(int size, out int gridX, out int blockX)    
        {
            if (size <= _maxThreads)
            {
                gridX = 1;
                blockX = size;
                return;
            }
            
            blockX = _maxThreads;
            gridX = (int)Math.Ceiling((double)size / _maxThreads);
        }

        public void LaunchKernel(string kernelName, int size, int sharedMemory, params object[] parameters)
        {
            CalcDimensions(size, out var gridx, out var blockX);
            LaunchKernel(kernelName, gridx, blockX, sharedMemory, parameters);
        }
    
        public void LaunchKernel(string kernelName, int gridSize, int blockSize, int sharedMemory, params object[] parameters)
        {
            var kernel = _kernels[kernelName];
            if (kernel == null)
                throw new ArgumentException($"Kernel with name [{kernelName}] not found");

            kernel.GridDimensions = gridSize;
            kernel.BlockDimensions = blockSize;    
            kernel.DynamicSharedMemory = (uint)sharedMemory;
            kernel.RunAsync(_context.Stream.Stream, parameters);
        }

    }
}
