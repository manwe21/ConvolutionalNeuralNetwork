namespace Training.Testers
{
    public class TestResult
    {
        public int TotalTests { get; set; }
        public int Successful { get; set; }
        public int Failed { get; set; }
        public float SuccessfulRatio { get; set; }

        public override string ToString()
        {
            return $"Total tests: {TotalTests}\n" +
                   $"Successful: {Successful}\n" +
                   $"Failed:{Failed}" +
                   $"\nSuccessfulRatio: {SuccessfulRatio}";
        }
    }
}