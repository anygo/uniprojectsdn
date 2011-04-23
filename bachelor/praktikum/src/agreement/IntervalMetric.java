package agreement;

/**
 * Interval metric. w(a,b) = (a-b)*(a-b)
 * 
 * @author sikoried
 */
public class IntervalMetric implements Metric
{
	public double weight(double a, double b)
	{
		return (a-b)*(a-b);
	}
	public String toString()
	{
		return "w(a,b) = (a-b)*(a-b)";
	}
}
