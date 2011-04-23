package agreement;

/**
 * Nominal metric. Used for binary decisions: Either match or not. Can
 * be inverted to result in 0 when a and b match.
 * 
 * @author sikoried
 */
public class NominalMetric implements Metric
{
	private boolean invert = false;
	
	/**
	 * Construct a standard nominal metric. a == b <=> w = 1
	 */
	public NominalMetric()
	{
	}
	
	/**
	 * Construct a nominal metric with optional invertation.
	 */
	public NominalMetric(boolean invert)
	{
		this.invert = invert;
	}
	public double weight(double a, double b)
	{
		double w = 0;
		if (a == b)
			w = 1;
		else
			w = 0;
		
		if (invert)
			return 1 - w;
		else
			return w;
	}
	public String toString()
	{
		return "w(a,b) = 1 <=> a != b; w(a,b) = 0 <=> a == b";
	}
}
