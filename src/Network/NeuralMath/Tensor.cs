using System.Text;
using Network.NeuralMath.Functions.ActivationFunctions;
using Network.NeuralMath.Functions.LossFunctions;

namespace Network.NeuralMath
{
    public abstract class Tensor
    {
        public TensorStorage Storage { get; }

        protected Tensor(TensorStorage storage)
        {
            Storage = storage;
        }
        
        public float this[int i]
        {
            get => Storage.Get(i);
            set => Storage.Set(i, value);
        }
        
        public float this[int i, int j]
        {
            get => Storage.Get(i, j);
            set => Storage.Set(i, j, value);
        }
        
        public float this[int c, int i, int j]
        {
            get => Storage.Get(c, i, j);
            set => Storage.Set(c, i, j, value);
        }
        
        public float this[int b, int c, int i, int j]
        {
            get => Storage.Get(b, c, i, j);
            set => Storage.Set(b, c, i, j, value);    
        }   
        
        public int Batch => Storage.Batch;
        public int Channels => Storage.Channels;
        public int Height => Storage.Height;
        public int Width => Storage.Width;
        
        public int Size => Storage.Size;

        public abstract void Dot2D(Tensor tensor, Tensor result);
        
        public abstract void DotTransA2D(Tensor tensor, Tensor result);
        
        public abstract void Transpose2D(Tensor result);
            
        //result tensor: shape = 1x1x1x2 ([0] - max value, [1] - max value index)
        public abstract void Max(Tensor result);

        public abstract void Sum(Tensor tensor); 
        
        //immutable version
        public abstract void Sum(Tensor tensor, Tensor result);

        public abstract void Fill(float value);
        
        //immutable version
        public abstract void Fill(float value, Tensor result);

        public abstract void Rotate180(Tensor result);
    
        public abstract void Img2Col(int kernelH, int kernelW, int stride, Tensor result);
        
        public abstract void Pad(int value, Tensor result);
        public abstract void PadDx(int value, Tensor dy, Tensor result);
                            
        public abstract void FullyConnectedDx(Tensor weights, Tensor dy, Tensor result);
        public abstract void FullyConnectedDw(Tensor dy, Tensor result);

        public void Convolution(Tensor filters, int stride, Tensor result)
        {
            var tensorType = result.GetType();
            var builder = TensorBuilder.OfType(tensorType);
            Convolution(filters, stride, 0, builder.Empty(), result);
        }
        
        public abstract void Convolution(Tensor filters, int stride, int padding, Tensor img2ColBuffer, Tensor result);

        public void ConvolutionDx(Tensor filters, Tensor dy, Tensor dx)
        {
            var tensorType = dx.GetType();
            var builder = TensorBuilder.OfType(tensorType);
            ConvolutionDx(filters, dy, builder.Empty(), builder.Empty(), builder.Empty(), builder.Empty(), dx);
        }
            
        public abstract void ConvolutionDx(Tensor filters, Tensor dy, Tensor paddingBuffer, Tensor img2ColBuffer, Tensor filters2DBuffer, Tensor dot2DBuffer, Tensor dx);

        public void ConvolutionDw(Tensor filters, Tensor dy, int stride, Tensor dw)
        {
            var tensorType = dw.GetType();
            var builder = TensorBuilder.OfType(tensorType);
            var img2Col = TensorBuilder.OfType(tensorType).Empty();
            Img2Col(filters.Height, filters.Width, stride, img2Col);
            ConvolutionDw(filters, dy, builder.Empty(), builder.Empty(), img2Col, dw);
        }
        
        public abstract void ConvolutionDw(Tensor filters, Tensor dy, Tensor dy2DBuffer, Tensor dot2DBuffer, Tensor img2ColX, Tensor dw);    
        
        public abstract void MaxPool(int poolSize, int stride, Tensor result, Tensor indexes);
        public abstract void MaxPoolDx(Tensor dy, Tensor maxIndexes, Tensor result);            
    
        public abstract void Activation(IFunction function, Tensor result);
        public abstract void ActivationDx(IFunction function, Tensor dy, Tensor dx);

        public void Softmax(Tensor result)
        {
            Softmax(result, TensorBuilder.OfType(result.GetType()).Empty());    
        }
        
        public abstract void Softmax(Tensor result, Tensor maxBuffer);        
        public abstract void SoftmaxDx(Tensor dy, Tensor dx);    

        public abstract void Loss(Tensor correct, ILossFunction lossFunction, Tensor loss);
        public abstract void LossDerivative(Tensor correct, ILossFunction lossFunction, Tensor dy);
    
        public abstract void ToFlatten(Tensor tensor);
        public abstract void FlattenDx(Tensor dy, Tensor dx);

        public static Shape GetConvolutionalShape(Shape input, Shape filters, int stride, int padding)
        {
            return new Shape
            (
                input.Dimensions[0],
                filters.Dimensions[0],
                (input.Dimensions[2] - filters.Dimensions[2] + 2 * padding) / stride + 1,
                (input.Dimensions[3] - filters.Dimensions[3] + 2 * padding) / stride + 1
            );
        }

        public static Shape GetPoolingShape(Shape input, int poolSize, int stride)
        {
            return new Shape
            (
                input.Dimensions[0],
                input.Dimensions[1],
                (input.Dimensions[2] - poolSize) / stride + 1,
                (input.Dimensions[3] - poolSize) / stride + 1
            );
        }

        public static Shape GetFlattenShape(Shape input)
        {
            return new Shape(1, 1, 1, input.Size);
        }

        public static Shape GetImg2ColShape(Shape input, int kernelW, int kernelH, int stride)
        {
            return new Shape
            (
                input.Dimensions[0],
                1,
                kernelH * kernelW * input.Dimensions[1],
                ((input.Dimensions[2] - kernelH) / stride + 1) * ((input.Dimensions[3] - kernelW) / stride + 1)
            );
        }

        public static Shape GetPaddingShape(Shape input, int padding)
        {
            return new Shape
            (
                input.Dimensions[0],
                input.Dimensions[1],
                input.Dimensions[2] + padding * 2,
                input.Dimensions[3] + padding * 2
            );
        }

        public static Shape GetDot2DShape(Shape a, Shape b)
        {
            return new Shape(a.Dimensions[0], 1, a.Dimensions[2], b.Dimensions[3]);
        }

        public static Shape GetDot2DTransAShape(Shape a, Shape b)
        {
            return new Shape(a.Dimensions[0], 1, a.Dimensions[3], b.Dimensions[3]);
        }

        public override string ToString()
        {
            StringBuilder sb = new StringBuilder();
            for (int b = 0; b < Batch; b++)
            {
                for (int h = 0; h < Height; h++)
                {
                    for (int c = 0; c < Channels; c++)
                    {
                        for (int w = 0; w < Width; w++)
                        {
                            sb.Append($"{this[b, c, h, w]} ");
                        }
                        sb.Append("\t\t");
                    }
                    sb.Append("\n");
                }

                sb.Append("\n\n");
            }
            return sb.ToString();
        }
        
    }    
}
