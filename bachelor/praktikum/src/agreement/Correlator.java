package agreement;

import java.io.*;
import java.util.ArrayList;
import java.util.Collections;

/**
 * Correlate two random variables (double arrays). Available 
 * correlation measures: Spearman rho, Pearson r
 * 
 * @author sikoried
 */
public final class Correlator
{
	/**
	 * Compute Pearson's correlation coefficient r
	 * @param a Random variable, double array
	 * @param b Random variable, double array
	 * @return -1 <= r <= 1
	 * @throws Exception if array sized don't match.
	 * @author sikoried
	 */
	public static double pearsonCorrelation(double [] a, double [] b)
		throws Exception
	{
		if (a.length != b.length) 
			throw new Exception ("DoubleArrayCorrelator: Array length not equal: " + a.length + " <> " + a.length);
		
		double meanone = 0;
		double meantwo = 0;
		
		for (int i = 0; i < a.length; i++)
		{
			meanone += a[i];
			meantwo += b[i];
		}
		meanone /= a.length;
		meantwo /= b.length;
		double sumone = 0;
		double sumtwo = 0;
		double sumthree = 0;
		for (int i=0; i < b.length;i++){
			sumone += (a[i] - meanone) * (b[i] - meantwo);
			sumtwo += (a[i] - meanone) * (a[i] - meanone);
			sumthree += (b[i] - meantwo) * (b[i] - meantwo);
		}
		return sumone / (Math.sqrt(sumtwo)*Math.sqrt(sumthree));
	}
	
	/**
	 * Compute Spearman's rank correlation rho.
	 * @param a Random variable, double array
	 * @param b Random variable, double array
	 * @return -1 <= rho <= 1
	 * @throws Exception if array sizes don't match
	 * @author sikoried
	 */
	public static double spearmanCorrelation(double [] a, double [] b)
		throws Exception
	{
		if (a.length != b.length)
			throw new Exception ("DoubleArrayCorrelator: Array length not equal: " + a.length + " <> " + b.length);
		
		a = valsToRank(a);
		b = valsToRank(b);
		
		double rho = 0;
		
		for (int i = 0; i < a.length; ++i)
			rho += ((a[i] - b[i])*(a[i] - b[i]));
		
		rho = 1. - 6 * rho / (a.length * (a.length*a.length - 1));
		
		return rho;
	}
	private static double [] valsToRank(double [] a)
	{
		class Pair implements Comparable<Pair>
		{
			public double val;
			public int index;
			public Pair(int index, double val)
			{
				this.index = index;
				this.val = val;
			}
			public int compareTo(Pair p)
			{
				return (int)Math.signum(this.val - p.val);
			}
		}
		
		ArrayList<Pair> data = new ArrayList<Pair>();
		for (int i = 0; i < a.length; ++i)
			data.add(new Pair(i, a[i]));
		Collections.sort(data);
		
		int [] map = new int [a.length];
		double [] sorted = new double [a.length];		
		double [] ranks = new double [a.length];

		for (int i = 0; i < a.length; ++i)
		{
			sorted[i] = data.get(i).val;
			map[i] = data.get(i).index;
		}
		
		for (int i = 0; i < sorted.length; ++i)
		{
			int cnt = 0;
			double r = i+1;
			// accumulate for average rank
			while (i+1 < sorted.length && sorted[i+1] == sorted[i])
			{
				cnt++;
				i++;
				r += (i+1);
			}
			r /= (cnt+1);
			// set average rank backwards
			while (cnt >= 0)
			{
				ranks[i-cnt] = r;
				cnt--;
			}
		}
		
		for (int i = 0; i < a.length; ++i)
			sorted[map[i]] = ranks[i];
		
		return sorted;
	}
	
	public final static String synopsis = 
		"usage: Correlator datafile1 datafile2 <datafile2 ...>";
	
	public static void main(String [] args)
		throws Exception
	{
		
		if (args.length < 2)
		{
			System.out.println(Correlator.synopsis);
			System.exit(0);
		}
		
		double [][] data = new double [args.length][];

		// read in data:
		for (int i = 0; i < args.length; ++i)
		{
			try 
			{
				BufferedReader in = null;
				if (args[i].equals("-")) {
					in = new BufferedReader(new InputStreamReader(System.in));
					args[i] = "STDIN";
				}
				else
					in = new BufferedReader(new FileReader(args[i]));
				ArrayList<Double> vals = new ArrayList<Double>();
				String l;
			    while ( (l = in.readLine()) != null )
					vals.add(new Double(l));
				in.close();

				data[i] = new double [vals.size()];
				for (int j = 0; j < vals.size(); ++j)
					data[i][j] = vals.get(j).doubleValue();
			}
			catch (Exception e)
			{
				System.out.println(e);
				e.printStackTrace();
			}
		}
		
		// compute each correlation
		for (int i = 0; i < data.length; ++i) {
			for (int j = i+1; j < data.length; ++j) {
				System.out.println(args[i] + "<>" + args[j] + ": r   = " + pearsonCorrelation(data[i], data[j]));
				System.out.println(args[i] + "<>" + args[j] + ": rho = " + spearmanCorrelation(data[i], data[j]));
			}
		}
		
		// compute one-vs-rest correlation using truth-sets
//		for (int i = 0; i < data.length; ++i) {
//			boolean [] seed = new boolean [data.length];
//			for (int j = 0; j < data.length; ++j) if (i != j) seed[j] = true; seed[i] = false;
//			GoldenTruth.TruthSet ts = new GoldenTruth.TruthSet(data[i], data, seed, GoldenTruth.EXPANSION_PEARSON);
//			System.out.println(args[i] + "<>rest: r   = " + ts.getPearson());
//			System.out.println(args[i] + "<>rest: rho = " + ts.getSpearman());
//			System.out.println(args[i] + "<>rest: RMSE = " + Math.sqrt(ts.getMeanSquareError()));
//		}
	}
}
