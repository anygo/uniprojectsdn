package agreement;

import java.io.*;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.Map;
import java.util.Iterator;


public final class Alpha implements AgreementMeasure
{
	private Metric metric;
	public Alpha(Metric metric)
	{
		this.metric = metric;
	}
	public Metric getMetric()
	{
		return metric;
	}
	/**
	 * Calculate Krippendorff's Alpha. Missing ratings have to be Double.MAX_VALUE
	 * 
	 * @param data data[rater][recording] = mark
	 * @param metric Weighting metric to use.
	 * @author sikoried
	 * @return alpha
	 */
	public double agreement(double [][] data) 
	{
		int ns = data[0].length; // number of subjects
		int nc = 0; // number of classes
		int nr = 0; // number of ratings
		int ne = data.length; // number of experts

		int [] m = new int [ns]; // ratings per subject
		
		// ensure that class id are consistent with value in case of nonnominal metrics
		ArrayList<Double> flat = new ArrayList<Double>();
		for (int i = 0; i < data.length; ++i)
			for (int j = 0; j < data[i].length; ++j)
				flat.add(data[i][j]);
		Collections.sort(flat);
		
		HashMap<Double, Integer> classMap = new HashMap<Double, Integer>();
		for (double d : flat) {
			if (!classMap.containsKey(d))
				classMap.put(d, new Integer(nc++));
		}
		
		// initialize nc, nr, ne, m
		for (int i = 0; i < ne; ++i)
		{
			for (int j = 0; j < ns; ++j)
			{
				if (data[i][j] == Double.MAX_VALUE)
					continue;
				m[j]++;
				nr++;
			}
		}
		
		// construct coincidence matrix
		double [][] coincidence = new double [nc][nc];
		
		// for every subject...
		for (int s = 0; s < ns; ++s)
		{
			// ...check all pairs
			HashMap <Pair, Integer> lc = new HashMap<Pair, Integer>(); // local coincidences
			for (int i = 0; i < ne; ++i)
			{
				if (data[i][s] == Double.MAX_VALUE)
					continue;
				
				int cid1 = classMap.get(data[i][s]);
				
				for (int j = 0; j < ne; ++j)
				{
					if (i == j)
						continue;
					
					if (data[j][s] == Double.MAX_VALUE)
						continue;
					
					int cid2 = classMap.get(data[j][s]);
					
					Pair key = new Pair(cid1, cid2);
					
					if (lc.containsKey(key))
						lc.put(key, 1 + lc.get(key));
					else
						lc.put(key, 1);
				}
			}
			
			// transfer local coincidences to coincidence matrix
			Iterator<Map.Entry<Pair, Integer>> it = lc.entrySet().iterator();
			while (it.hasNext())
			{
				Map.Entry<Pair, Integer> e = it.next();
				int cid1 = e.getKey().a;
				int cid2 = e.getKey().b;
				coincidence[cid1][cid2] += ((double)e.getValue()/(m[s]-1));
			}
			
		}
		
//		for (int i = 0; i < coincidence.length; ++i) {
//			for (int j = 0; j < coincidence[i].length; ++j)
//				System.out.print(coincidence[i][j] + "\t");
//			System.out.println("");
//		}
		
		// count occurrences
		double na = 0;
		double [] occurrence = new double [nc];
		
		for (int i = 0; i < nc; i++)
		{
			for (int j = 0; j < nc; j++)
			{
				double ci = coincidence[i][j];
				if (ci == Double.MAX_VALUE)
					continue;
				occurrence[i] += ci;
				na += ci;
			}
		}
		
		double d_o = 0;
		for (int i = 0; i < nc; ++i)
		{
			for (int j = i+1; j < nc; ++j)
			{
				d_o += metric.weight(i+1, j+1) * coincidence[i][j];
			}
		}
		d_o *= (na-1);
		
		double d_e = 0;
		for (int i = 0; i < nc; ++i)
		{
			for (int j = i+1; j < nc; ++j)
			{
				d_e += metric.weight(i+1, j+1) * occurrence[i] * occurrence[j];
			}
		}

		return 1. - d_o/d_e;
	}
	
	private static final class Pair
	{
		public int a;
		public int b;
		public Pair(int a, int b)
		{
			this.a = a;
			this.b = b;
		}
		public boolean equals(Pair p)
		{
			return a == p.a && b == p.b;
		}
		public boolean equals(Object o)
		{
			if (o instanceof Pair)
				return equals((Pair)o);
			else
				return false;
		}
	}
	
	public static final String synopsis = 
		"Usage: Alpha rater1 rater2 <rater3 ...>";
	
	public static void main(String[] args)
	{
//		double c = Double.MAX_VALUE;
//		double data1 [][] = {
//				{ 1, 2, 3, 3, 2, 1, 4, 1, 2, c, c, c },
//				{ 1, 2, 3, 3, 2, 2, 4, 1, 2, 5, c, c },
//				{ c, 3, 3, 3, 2, 3, 4, 2, 2, 5, 1, 3 },
//				{ 1, 2, 3, 3, 2, 4, 4, 1, 2, 5, 1, c }
//		};
		
		if (args.length < 2)
		{
			System.out.println(Alpha.synopsis);
			System.exit(0);
		}

		double [][] data = new double [args.length][];

		// read in data:
		for (int i = 0; i < args.length; ++i)
		{
			try 
			{
				BufferedReader in = new BufferedReader(args[i].equals("-") ? new InputStreamReader(System.in) : new FileReader(args[i]));
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
		
//		for (int i = 0; i < data[0].length; ++i) {
//			for (int j = 0; j < data.length; ++j)
//				System.out.print(data[j][i] + "\t");
//			System.out.println("");
//		}

		Alpha alpha1 = new Alpha(new NominalMetric(true));
		Alpha alpha2 = new Alpha(new IntervalMetric());
		Alpha alpha3 = new Alpha(new RatioMetric());
		
		System.out.println("alpha nominal = " + alpha1.agreement(data));
		System.out.println("alpha interval = " + alpha2.agreement(data));
		System.out.println("alpha ratio = " + alpha3.agreement(data));
	}
}
