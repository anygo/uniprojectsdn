package agreement;

/**
 * Metrics proposed by Cicchetti in Cicchetti76:AIR. Possible modes: ABSOLUTE and SQUARE
 * 
 * @author sikoried
 */
public class CicchettiMetric implements Metric
{
	double c = 1;
	private int mode;
	
	public static final int ABSOLUTE = 0;
	public static final int SQUARE = 1;
	
	public CicchettiMetric(int mode)
	{
		if (mode == ABSOLUTE)
			this.mode = ABSOLUTE;
		else if (mode == SQUARE)
			this.mode = SQUARE;
		else
			this.mode = ABSOLUTE;
	}
	public double weight(double a, double b)
	{
		if (mode == ABSOLUTE)
			return (a == b ? 1. : (1. - Math.abs((a-b)/(c-1.))));
		else
			return (a == b ? 1. : (1. - Math.pow((a-b)/(c-1.), 2)));
	}
	public String toString()
	{
		if (mode == ABSOLUTE)
			return "(a == b ? 1. : (1. - Math.abs((a-b)/(" + c + "))))";
		else
			return "(a == b ? 1. : (1. - Math.pow((a-b)/(" + c + "), 2)))";
	}
}
