﻿using System;
using ManagedCuda;

namespace Network.NeuralMath
{
    public abstract class TensorStorage
    {
        private Shape _shape;
        
        public abstract float[] Array { get; }
        
        public bool IsMemoryAllocated { get; protected set; }

        public Shape Shape
        {
            get => _shape;
            set
            {
                _shape = value;
                Batch = _shape.Dimensions[0];
                Channels = _shape.Dimensions[1];
                Height = _shape.Dimensions[2];
                Width = _shape.Dimensions[3];
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

        public abstract void SetData(float[] data);    

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