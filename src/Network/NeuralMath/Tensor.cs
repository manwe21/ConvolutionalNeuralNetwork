using System;
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
        
        public void Dot2D(Tensor b, Tensor c)
        {
            if (b == null) throw new ArgumentNullException(nameof(b));
            if (c == null) throw new ArgumentNullException(nameof(c));
            
            c.Storage.AllocateMemory(GetDot2DShape(Storage.Shape, b.Storage.Shape));
            DoDot2D(b, c);
        }

        public void Dot2D(Tensor b, int hA, int wA, int hB, int wB, Shape resultShape, Tensor c)
        {
            if (b == null) throw new ArgumentNullException(nameof(b));
            if (c == null) throw new ArgumentNullException(nameof(c));
            
            c.Storage.AllocateMemory(resultShape ?? new Shape(1, 1, hA, wB));
            DoDot2D(b, hA, wA, hB, wB, resultShape, c);
        }
        
        public void Transpose2D(Tensor result)
        {
            if (result == null) throw new ArgumentNullException(nameof(result));
            
            result.Storage.AllocateMemory(new Shape(1, 1, Width, Height));
            DoTranspose2D(result);
        }
        
        //result tensor: shape = Bx1x1x2 ([0] - max value, [1] - max value index(in batch))
        public void Max(Tensor result)
        {
            if (result == null) throw new ArgumentNullException(nameof(result));
            
            result.Storage.AllocateMemory(new Shape(Batch, 1, 1, 2));
            FindMax(result);
        }
        
        public void Average(Tensor result)
        {
            if (result == null) throw new ArgumentNullException(nameof(result));
            
            result.Storage.AllocateMemory(new Shape(Batch, 1, 1, 1));
            FindAverage(result);
        }
        
        public void Sum(Tensor tensor)
        {
            if (tensor == null) throw new ArgumentNullException(nameof(tensor));
            
            DoSum(tensor);
        }
        
        //immutable version
        public void Sum(Tensor tensor, Tensor result)
        {
            if (tensor == null) throw new ArgumentNullException(nameof(tensor));
            if (result == null) throw new ArgumentNullException(nameof(result));
            
            result.Storage.AllocateMemory(Storage.Shape.GetCopy());
            DoSum(tensor, result);
        }
        
        //immutable version
        public void Fill(float value, Tensor result)
        {
            if (result == null) throw new ArgumentNullException(nameof(result));
            
            result.Storage.AllocateMemory(Storage.Shape.GetCopy());
            DoFilling(value, result);
        }
        
        public void Rotate180(Tensor result)
        {
            if (result == null) throw new ArgumentNullException(nameof(result));
            
            result.Storage.AllocateMemory(Storage.Shape.GetCopy());
            DoRotate180(result);
        }
        
        public void Im2Col(int kernelH, int kernelW, int stride, Tensor result)
        {
            if (result == null) throw new ArgumentNullException(nameof(result));
            
            result.Storage.AllocateMemory(GetIm2ColShape(Storage.Shape, kernelW, kernelH, stride));
            DoIm2Col(kernelH, kernelW, stride, result);
        }
        
        public void Col2Im(Shape outShape, Tensor result)
        {
            if (outShape == null) throw new ArgumentNullException(nameof(outShape));
            if (result == null) throw new ArgumentNullException(nameof(result));
            
            result.Storage.AllocateMemory(outShape);
            DoCol2Im(outShape, result);
        }
        
        public void Pad(int value, Tensor result)
        {
            if (result == null) throw new ArgumentNullException(nameof(result));
            
            result.Storage.AllocateMemory(GetPaddingShape(Storage.Shape, value));
            DoPad(value, result);
        }
        
        public void PadDx(int value, Tensor dy, Tensor dx)
        {
            if (dy == null) throw new ArgumentNullException(nameof(dy));
            if (dx == null) throw new ArgumentNullException(nameof(dx));
            
            dx.Storage.AllocateMemory(new Shape(dy.Batch, dy.Channels, dy.Height - 2 * value, dy.Width - 2 * value));
            DoPadDx(value, dy, dx);
        }
        
        public void MaxPool(int poolSize, int stride, Tensor result, Tensor indexes)
        {
            if (result == null) throw new ArgumentNullException(nameof(result));
            if (indexes == null) throw new ArgumentNullException(nameof(indexes));
            
            result.Storage.AllocateMemory(GetPoolingShape(Storage.Shape, poolSize, stride));
            indexes.Storage.AllocateMemory(Shape.ForVector(result.Size));
            DoMaxPool(poolSize, stride, result, indexes);
        }
        
        public void MaxPoolDx(Tensor dy, Tensor maxIndexes, Tensor dx)
        {
            if (dy == null) throw new ArgumentNullException(nameof(dy));
            if (maxIndexes == null) throw new ArgumentNullException(nameof(maxIndexes));
            if (dx == null) throw new ArgumentNullException(nameof(dx));
            
            dx.Storage.AllocateMemory(Storage.Shape.GetCopy());
            DoMaxPoolDx(dy, maxIndexes, dx);
        }
        
        public void Activation(IFunction function, Tensor result)
        {
            if (function == null) throw new ArgumentNullException(nameof(function));
            if (result == null) throw new ArgumentNullException(nameof(result));
            
            result.Storage.AllocateMemory(Storage.Shape.GetCopy());
            DoActivation(function, result);
        }
        
        public void ActivationDx(IFunction function, Tensor dy, Tensor dx)
        {
            if (function == null) throw new ArgumentNullException(nameof(function));
            if (dy == null) throw new ArgumentNullException(nameof(dy));
            if (dx == null) throw new ArgumentNullException(nameof(dx));
            
            dx.Storage.AllocateMemory(Storage.Shape.GetCopy());
            DoActivationDx(function, dy, dx);
        }
        
        public void Softmax(Tensor result, Tensor maxBuffer)
        {
            if (result == null) throw new ArgumentNullException(nameof(result));
            if (maxBuffer == null) throw new ArgumentNullException(nameof(maxBuffer));
            
            result.Storage.AllocateMemory(Storage.Shape.GetCopy());
            DoSoftmax(result, maxBuffer);
        }
        
        public void SoftmaxDx(Tensor dy, Tensor dx)
        {
            if (dy == null) throw new ArgumentNullException(nameof(dy));
            if (dx == null) throw new ArgumentNullException(nameof(dx));
            
            dx.Storage.AllocateMemory(Storage.Shape.GetCopy());
            DoSoftmaxDx(dy, dx);
        }
        
        public void Loss(Tensor correct, ILossFunction lossFunction, Tensor loss)
        {
            if (correct == null) throw new ArgumentNullException(nameof(correct));
            if (lossFunction == null) throw new ArgumentNullException(nameof(lossFunction));
            if (loss == null) throw new ArgumentNullException(nameof(loss));
            
            loss.Storage.AllocateMemory(new Shape(Batch, 1, 1, 1));
            DoLoss(correct, lossFunction, loss);
        }
        
        public void LossDerivative(Tensor correct, ILossFunction lossFunction, Tensor dy)
        {
            if (correct == null) throw new ArgumentNullException(nameof(correct));
            if (lossFunction == null) throw new ArgumentNullException(nameof(lossFunction));
            if (dy == null) throw new ArgumentNullException(nameof(dy));
            
            dy.Storage.AllocateMemory(Storage.Shape.GetCopy());
            DoLossDerivative(correct, lossFunction, dy);
        }
        
        public void ToFlatten(Tensor result)
        {
            if (result == null) throw new ArgumentNullException(nameof(result));
            
            result.Storage.AllocateMemory(GetFlattenShape(Storage.Shape));
            DoFlattening(result);
        }
        
        public void FlattenDx(Tensor dy, Tensor dx)
        {
            if (dy == null) throw new ArgumentNullException(nameof(dy));
            if (dx == null) throw new ArgumentNullException(nameof(dx));
            
            dx.Storage.AllocateMemory(Storage.Shape.GetCopy());
            DoFlatteningDx(dy, dx);
        }
        
        public void To2DByRows(Tensor result)
        {
            if (result == null) throw new ArgumentNullException(nameof(result));
            
            result.Storage.AllocateMemory(Get2DByRowsShape(Storage.Shape));
            Do2DReshapingByRows(result);
        }
        
        public void To2DByColumns(Tensor result)
        {
            if (result == null) throw new ArgumentNullException(nameof(result));
            
            result.Storage.AllocateMemory(Get2DByColumnsShape(Storage.Shape));
            Do2DReshapingByColumns(result);
        }
        
        public void ReshapeForBatches(Shape resultShape, Tensor result)
        {
            if (resultShape == null) throw new ArgumentNullException(nameof(resultShape));
            if (result == null) throw new ArgumentNullException(nameof(result));
            
            result.Storage.AllocateMemory(resultShape);
            DoReshapingForBatches(resultShape, result);
        }
        
        public void FullyConnectedDx(Tensor weights, Tensor dy, Tensor transBuffer, Tensor dx)
        {
            if (weights == null) throw new ArgumentNullException(nameof(weights));
            if (dy == null) throw new ArgumentNullException(nameof(dy));
            if (transBuffer == null) throw new ArgumentNullException(nameof(transBuffer));
            if (dx == null) throw new ArgumentNullException(nameof(dx));
            
            weights.Transpose2D(transBuffer);
            dy.Dot2D(transBuffer, dy.Batch, dy.Width, transBuffer.Height, transBuffer.Width, Storage.Shape, dx);
        }

        public void FullyConnectedDw(Tensor dy, Tensor transBuffer, Tensor dw)
        {
            if (dy == null) throw new ArgumentNullException(nameof(dy));
            if (transBuffer == null) throw new ArgumentNullException(nameof(transBuffer));
            if (dw == null) throw new ArgumentNullException(nameof(dw));
            
            var initialShape = Storage.Shape;
            Storage.Shape = new Shape(1, 1, Batch, Width);
            this.Transpose2D(transBuffer);
            transBuffer.Dot2D(dy, transBuffer.Height, transBuffer.Width, dy.Batch, dy.Width, dw.Storage.Shape, dw);
            Storage.Shape = initialShape;
        }

        public void Convolution(Tensor filters, int stride, Tensor img2ColBuffer, Tensor dotBuffer, Tensor result)
        {
            if (filters == null) throw new ArgumentNullException(nameof(filters));
            if (img2ColBuffer == null) throw new ArgumentNullException(nameof(img2ColBuffer));
            if (dotBuffer == null) throw new ArgumentNullException(nameof(dotBuffer));
            if (result == null) throw new ArgumentNullException(nameof(result));
            
            result.Storage.AllocateMemory(GetConvolutionalShape(Storage.Shape, filters.Storage.Shape, stride, 0));
            
            this.Im2Col(filters.Height, filters.Width, stride, img2ColBuffer);
            filters.Dot2D(img2ColBuffer,
                filters.Batch,
                filters.Channels * filters.Height * filters.Width,
                img2ColBuffer.Height,
                img2ColBuffer.Width,
                null,
                dotBuffer);
            dotBuffer.Col2Im(result.Storage.Shape, result);
        }

        public void ConvolutionDx(Tensor filters, Tensor dy, Tensor paddingBuffer, Tensor img2ColBuffer, Tensor filters2DBuffer, Tensor rotationBuffer, Tensor dot2DBuffer, Tensor dx)
        {
            if (filters == null) throw new ArgumentNullException(nameof(filters));
            if (dy == null) throw new ArgumentNullException(nameof(dy));
            if (paddingBuffer == null) throw new ArgumentNullException(nameof(paddingBuffer));
            if (img2ColBuffer == null) throw new ArgumentNullException(nameof(img2ColBuffer));
            if (filters2DBuffer == null) throw new ArgumentNullException(nameof(filters2DBuffer));
            if (rotationBuffer == null) throw new ArgumentNullException(nameof(rotationBuffer));
            if (dot2DBuffer == null) throw new ArgumentNullException(nameof(dot2DBuffer));
            if (dx == null) throw new ArgumentNullException(nameof(dx));
            
            dx.Storage.AllocateMemory(Storage.Shape.GetCopy());
            
            dy.Pad(Width - dy.Width, paddingBuffer);
            paddingBuffer.Im2Col(filters.Height, filters.Width, 1, img2ColBuffer);
            filters.Rotate180(rotationBuffer);
            rotationBuffer.To2DByRows(filters2DBuffer);
            filters2DBuffer.Dot2D(img2ColBuffer, dot2DBuffer);
            dot2DBuffer.ReshapeForBatches(Storage.Shape, dx);
        }
        
        public void ConvolutionDw(Tensor filters, Tensor dy, Tensor dy2DBuffer, Tensor dotBuffer, Tensor img2ColX, Tensor dw)
        {
            if (filters == null) throw new ArgumentNullException(nameof(filters));
            if (dy == null) throw new ArgumentNullException(nameof(dy));
            if (dy2DBuffer == null) throw new ArgumentNullException(nameof(dy2DBuffer));
            if (dotBuffer == null) throw new ArgumentNullException(nameof(dotBuffer));
            if (img2ColX == null) throw new ArgumentNullException(nameof(img2ColX));
            if (dw == null) throw new ArgumentNullException(nameof(dw));
            
            dw.Storage.AllocateMemory(Get2DByRowsShape(filters.Storage.Shape));
            
            dy.To2DByColumns(dy2DBuffer);
            img2ColX.Dot2D(dy2DBuffer, dotBuffer);
            dotBuffer.Transpose2D(dw);
            dw.Storage.Shape = filters.Storage.Shape;
        }

        protected abstract void DoDot2D(Tensor b, Tensor c);

        protected abstract void DoDot2D(Tensor b, int hA, int wA, int hB, int wB, Shape resultShape, Tensor c);

        protected abstract void DoTranspose2D(Tensor result);

        protected abstract void FindMax(Tensor result);

        protected abstract void FindAverage(Tensor result);

        protected abstract void DoSum(Tensor tensor);

        protected abstract void DoSum(Tensor tensor, Tensor result);

        public abstract void Fill(float value);

        protected abstract void DoFilling(float value, Tensor result);

        protected abstract void DoRotate180(Tensor result);

        protected abstract void DoIm2Col(int kernelH, int kernelW, int stride, Tensor result);

        protected abstract void DoCol2Im(Shape outShape, Tensor result);

        protected abstract void DoPad(int value, Tensor result);

        protected abstract void DoPadDx(int value, Tensor dy, Tensor dx);

        protected abstract void DoMaxPool(int poolSize, int stride, Tensor result, Tensor indexes);

        protected abstract void DoMaxPoolDx(Tensor dy, Tensor maxIndexes, Tensor dx);

        protected abstract void DoActivation(IFunction function, Tensor result);

        protected abstract void DoActivationDx(IFunction function, Tensor dy, Tensor dx);

        protected abstract void DoSoftmax(Tensor result, Tensor maxBuffer);

        protected abstract void DoSoftmaxDx(Tensor dy, Tensor dx);

        protected abstract void DoLoss(Tensor correct, ILossFunction lossFunction, Tensor loss);

        protected abstract void DoLossDerivative(Tensor correct, ILossFunction lossFunction, Tensor dy);

        protected abstract void DoFlattening(Tensor tensor);

        protected abstract void DoFlatteningDx(Tensor dy, Tensor dx);

        protected abstract void Do2DReshapingByRows(Tensor result);
        
        protected abstract void Do2DReshapingByColumns(Tensor result);

        protected abstract void DoReshapingForBatches(Shape resultShape, Tensor result);

        public static Shape GetConvolutionalShape(Shape input, Shape filters, int stride, int padding)
        {
            return new Shape
            (
                input[0],
                filters[0],
                (input[2] - filters[2] + 2 * padding) / stride + 1,
                (input[3] - filters[3] + 2 * padding) / stride + 1
            );
        }

        public static Shape GetPoolingShape(Shape input, int poolSize, int stride)
        {
            return new Shape
            (
                input[0],
                input[1],
                (input[2] - poolSize) / stride + 1,
                (input[3] - poolSize) / stride + 1
            );
        }

        public static Shape GetFlattenShape(Shape input)
        {
            return new Shape(input[0], 1, 1, input.Size / input[0]);
        }

        public static Shape GetIm2ColShape(Shape input, int kernelW, int kernelH, int stride)
        {
            return new Shape
            (
                1,
                1,
                kernelH * kernelW * input[1],
                ((input[2] - kernelH) / stride + 1) * ((input[3] - kernelW) / stride + 1) * input[0]
            );
        }

        public static Shape GetPaddingShape(Shape input, int padding)
        {
            return new Shape
            (
                input[0],
                input[1],
                input[2] + padding * 2,
                input[3] + padding * 2
            );
        }

        public static Shape GetDot2DShape(Shape a, Shape b)
        {
            return new Shape(a[0], 1, a[2], b[3]);
        }

        public static Shape Get2DByColumnsShape(Shape input)
        {
            return new Shape(1, 1, input[0] * input[2] * input[3], input[1]);
        }

        public static Shape Get2DByRowsShape(Shape input)
        {
            return new Shape(1, 1, input[1], input[0] * input[2] * input[3]);
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
