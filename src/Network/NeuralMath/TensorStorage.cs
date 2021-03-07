using System;

namespace Network.NeuralMath
{
    public abstract class TensorStorage
    {
        private Shape _shape;
        
        public abstract float[] Data { get; set; }
        
        public bool IsMemoryAllocated { get; protected set; }

        public Shape Shape
        {
            get => _shape;
            set
            {
                if (_shape is null)
                    throw new ArgumentNullException(nameof(value));
                
                //can`t reshape tensor by shape with different size
                if(Shape.Size != value.Size)
                    throw new ArgumentException(nameof(value));
                

                _shape = value;
                Batch = _shape[0];
                Channels = _shape[1];
                Height = _shape[2];
                Width = _shape[3];
                Size = _shape.Size;

                Hw = Height * Width;
                Chw = Channels * Height * Width;
                
                Descriptor = new TensorDescriptor(Batch, Channels, Height, Width, Size);
            }
        }
        
        public int Batch { get; private set; }
        public int Channels { get; private set; }
        public int Height { get; private set; }
        public int Width { get; private set; }
        
        public int Size { get; private set; }
        
        public TensorDescriptor Descriptor { get; private set; }
        
        protected int Hw;
        protected int Chw;

        protected TensorStorage()
        {
            
        }

        protected TensorStorage(Shape shape)
        {
            AllocateMemory(shape);
        }

        protected TensorStorage(Shape shape, float[] data)
        {
            Data = data;
            Shape = shape;
        }

        public void Reshape(Shape shape)
        {
            if(Shape.Size != shape.Size)
                throw new ArgumentException(nameof(shape));

            Shape = shape;
        }

        public abstract void AllocateMemory(int size);

        public abstract void AllocateMemory(Shape shape);

        //abstract methods affect performance
        public abstract float Get(int i);

        public abstract float Get(int i, int j);

        public abstract float Get(int c, int i, int j);

        public abstract float Get(int b, int c, int i, int j);

        public abstract void Set(int i, float value);

        public abstract void Set(int i, int j, float value);

        public abstract void Set(int c, int i, int j, float value);

        public abstract void Set(int b, int c, int i, int j, float value);

    }
}
