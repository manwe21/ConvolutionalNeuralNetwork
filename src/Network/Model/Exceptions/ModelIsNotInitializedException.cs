using System;

namespace Network.Model.Exceptions
{
    public class ModelIsNotInitializedException : Exception
    {
        public ModelIsNotInitializedException() : base("Model is not initialized")
        {
            
        }
    }
}