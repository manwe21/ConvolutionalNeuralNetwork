using System;

namespace Network.NeuralMath.Cpu
{
    public class CpuStorage : TensorStorage
    {
        private float[] _array;
        
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
            get => _array;
            set
            {
                if (IsMemoryAllocated)
                {
                    if(_array.Length != value.Length)
                        throw new ArgumentException(nameof(value));
                    _array = value;
                }
                else
                {
                    _array = value;
                    Shape = new Shape(1, 1, 1, _array.Length);
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
            _array = new float[shape.Size];
            Shape = shape;
            IsMemoryAllocated = true;
        }

        public override float Get(int i)
        {
            return _array[i];
        }

        public override float Get(int i, int j)
        {
            return _array[i * Width + j];
        }

        public override float Get(int c, int i, int j)
        {
            return _array[c * Hw + i * Width + j];
        }

        public override float Get(int b, int c, int i, int j)
        {
            return _array[c * Hw + i * Width + j + b * Chw];
        }

        public override void Set(int i, float value)
        {
            _array[i] = value;
        }

        public override void Set(int i, int j, float value)
        {
            _array[i * Width + j] = value;
        }

        public override void Set(int c, int i, int j, float value)
        {
            _array[c * Hw + i * Width + j] = value;
        }

        public override void Set(int b, int c, int i, int j, float value)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         
        {
            _array[c * Hw + i * Width + j + b * Chw] = value;
        }
        
        
    }
}
