package agreement;

public abstract class Utils {
	/**
	 * Add a rater to a existing group of raters.
	 * @param data existing rater set
	 * @param rater new rater to add
	 * @return copy of data arry plus new rater
	 */
	public static double [][] addRater(double [][] data, double [] rater)
	{
		double [][] dnew = new double [data.length+1][rater.length];
		for (int i = 0; i < data.length; ++i)
			for (int j = 0; j < data[i].length; ++j)
				dnew[i][j] = data[i][j];
		for (int i = 0; i < data[0].length; ++i)
			dnew[data.length][i] = rater[i];
		return dnew;
	}
}
