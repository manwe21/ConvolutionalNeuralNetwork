using ManagedCuda;
using ManagedCuda.CudaBlas;

namespace Network.NeuralMath.Gpu
{
    public class GpuContext
    {
        private static GpuContext _context;
        public KernelManager KernelManager { get; }
        public static GpuContext Instance
        {
            get
            {
                if (_context == null)
                {
                    _context = new GpuContext();
                }
                return _context;
            }
        }
        
        public CudaStream Stream { get; }
        public CudaContext CudaContext { get; }
        public CudaBlas BlasContext { get; }
        public TensorMethods Methods { get; }
        
        private GpuContext()
        {
            CudaContext = new CudaContext(Global.CudaDeviceId);
            BlasContext = new CudaBlas();
            KernelManager = new KernelManager(this);
            Methods = new TensorMethods(this);
            Stream = new CudaStream();
        }

    }
}
