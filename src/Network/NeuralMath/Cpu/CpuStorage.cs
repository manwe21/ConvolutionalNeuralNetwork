using System;

namespace Network.NeuralMath.Cpu
{
    public class CpuStorage : TensorStorage
    {
        private float[] _data;
        
        public CpuStorage()
        {
        }    

        public CpuStorage(Shape shape) : base(shape)
        {

        }
        
        public CpuStorage(Shape shape, float[] data) : base(shape, data)
        {
            
        }

        public override float[] Data
        {
            get => _data;
            set
            {
                if (IsMemoryAllocated)
                {
                    if(_data.Length != value.Length)
                        throw new ArgumentException(nameof(value));
                    _data = value;
                }
                else
                {
                    _data = value;
                    Shape = new Shape(1, 1, 1, _data.Length);
                    IsMemoryAllocated = true;
                }
            }
        }

        public override void AllocateMemory(int size)
        {
            if(size <= 0)
                throw new ArgumentOutOfRangeException(nameof(size));
            
            AllocateMemory(new Shape(1, 1, 1, size));
        }
        
        public override void AllocateMemory(Shape shape)
        {
            if(IsMemoryAllocated)
                return;
            
            _data = new float[shape.Size];
            Shape = shape;
            IsMemoryAllocated = true;
        }

        public override float Get(int i)
        {
            return _data[i];
        }

        public override float Get(int i, int j)
        {
            return _data[i * Width + j];
        }

        public override float Get(int c, int i, int j)
        {
            return _data[c * Hw + i * Width + j];
        }

        public override float Get(int b, int c, int i, int j)
        {
            return _data[c * Hw + i * Width + j + b * Chw];
        }

        public override void Set(int i, float value)
        {
            _data[i] = value;
        }

        public override void Set(int i, int j, float value)
        {
            _data[i * Width + j] = value;
        }

        public override void Set(int c, int i, int j, float value)
        {
            _data[c * Hw + i * Width + j] = value;
        }

        public override void Set(int b, int c, int i, int j, float value)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         
        {
            _data[c * Hw + i * Width + j + b * Chw] = value;
        }
        
        
    }
}
