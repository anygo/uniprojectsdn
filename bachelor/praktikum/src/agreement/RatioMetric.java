package agreement;

/**
 * RatioMetric. A bit smoother than interval metric.
 * 
 * @author sikoried
 */
public class RatioMetric implements Metric
{
	public double weight(double a, double b)
	{
		return Math.pow((a-b)/(a+b), 2);
	}
	public String toString()
	{
		return "w(a,b) = Math.pow((a-b)/(a+b), 2)";
	}
}
