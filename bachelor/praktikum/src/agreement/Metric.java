package agreement;

/**
 * Metric to weight differences in decisions.
 * 
 * @author sikoried
 */
public interface Metric
{
	double weight(double a, double b);
	String toString();
}
