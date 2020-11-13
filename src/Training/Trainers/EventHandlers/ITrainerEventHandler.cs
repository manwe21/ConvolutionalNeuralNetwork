namespace Training.Trainers.EventHandlers
{
    public interface ITrainerEventHandler
    {
        void OnIterationFinished(IterationResult result);
        void OnEpochFinished(EpochResult result);
    }
}
