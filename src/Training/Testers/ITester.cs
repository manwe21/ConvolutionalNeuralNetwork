using Network.Model;

namespace Training.Testers
{
    public interface ITester
    {
        TestResult TestModel(INetwork network);
    }
}
