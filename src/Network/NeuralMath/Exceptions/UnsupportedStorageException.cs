using System;

namespace Network.NeuralMath.Exceptions
{
    public class UnsupportedStorageException : ArgumentException
    {
        public UnsupportedStorageException(string paramName)
            : base($"{ paramName } has unsupported storage type")
        {
            
        }
    }
}