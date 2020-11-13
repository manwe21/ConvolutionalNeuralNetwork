using System;
using System.Runtime.InteropServices;
using ManagedCuda;
using ManagedCuda.BasicTypes;

namespace Network.NeuralMath.Gpu
{
    public class GpuStorage : TensorStorage, IDisposable
    {
        public CudaDeviceVariable<float> DeviceStorage { get;  set; }
        public GpuContext Context { get; } = GpuContext.Instance;

        public GpuStorage()
        {
            
        }
        
        public GpuStorage(Shape shape) : base(shape)
        {
            
        }    
        
        public GpuStorage(Shape shape, float[] deviceStorage)
        {
            Shape = shape;
            SetData(deviceStorage);
        }

        public override float[] Array => DeviceStorage;
        
        public override void SetData(float[] data)
        {
            //allocate memory for host data
            IntPtr host = Marshal.AllocHGlobal(data.Length * sizeof(float));
            Marshal.Copy(data, 0, host, data.Length);
            
            //copy host data to device 
            var res = DriverAPINativeMethods.AsynchronousMemcpy_v2.cuMemcpyHtoDAsync_v2(DeviceStorage.DevicePointer, host,
                DeviceStorage.SizeInBytes, Context.Stream.Stream);
            if(res != CUResult.Success)
                throw new CudaException(res);
            
            //free host buffer
            Marshal.FreeHGlobal(host);
        }    

        public void SetData(CudaDeviceVariable<float> data)
        {
            if (IsMemoryAllocated)
            {
                if(DeviceStorage.Size != data.Size)
                    throw new ArgumentException(nameof(data));
                DeviceStorage = data;
            }
            else
            {
                DeviceStorage = data;
                Shape = new Shape(1, 1, 1, DeviceStorage.Size);
                IsMemoryAllocated = true;
            }
        }

        public override void AllocateMemory(int size)
        {
            if(size <= 0)
                throw new ArgumentException(nameof(size));
            
            AllocateMemory(new Shape(1, 1, 1, size));
        }

        public override void AllocateMemory(Shape shape)
        {
            DeviceStorage = new CudaDeviceVariable<float>(shape.Size);
            Shape = shape;
            IsMemoryAllocated = true;
        }

        public override float Get(int i)
        {
            return DeviceStorage[i];
        }

        public override float Get(int i, int j)
        {
            return DeviceStorage[i * Width + j];
        }

        public override float Get(int c, int i, int j)
        {
            return DeviceStorage[c * Hw + i * Width + j];
        }

        public override float Get(int b, int c, int i, int j)
        {
            return DeviceStorage[c * Hw + i * Width + j + b * Chw];
        }

        public override void Set(int i, float value)
        {
            DeviceStorage[i] = value;
        }

        public override void Set(int i, int j, float value)
        {
            DeviceStorage[i * Width + j] = value;
        }

        public override void Set(int c, int i, int j, float value)
        {
            DeviceStorage[c * Hw + i * Width + j] = value;
        }

        public override void Set(int b, int c, int i, int j, float value)
        {
            DeviceStorage[c * Hw + i * Width + j + b * Chw] = value;
        }

        public void FreeMemory()
        {
            Context.CudaContext.FreeMemory(DeviceStorage.DevicePointer);
        }

        public void Dispose()
        {
            DeviceStorage?.Dispose();
        }
    }
}
