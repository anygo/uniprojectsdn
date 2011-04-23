package util;

/**
 * This package provides commonly used distance measures, e.g. euclid, cityblock etc.
 * @author koried@icsi.berkeley.edu
 */
public abstract class Distances {
	
	public static double euclidean(double [] a, double [] b) {
		double dist = 0; 
		for (int i = 0; i < a.length; ++i)
			dist += Math.pow(a[i]-b[i], 2);
		return Math.sqrt(dist);
	}
	
	public static double manhattan(double [] a, double [] b)
		throws Exception {
		double dist = 0; 
		for (int i = 0; i < a.length; ++i)
			dist += Math.abs(a[i]-b[i]);
		return dist;
	}
}
