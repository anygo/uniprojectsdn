package agreement;

import java.io.*;
import java.util.*;

public class Histogram {
	/***
	 * Compute the histogram to a given list of values. The output array
	 * contains the values in ascending order
	 * @param data
	 * @return double['id'][0] = value, double['id'][1] = count where id = 0, ..., count(disjoint(data))
	 */
	public static double [][] histogram(double [] data) {
		HashMap<Double, Integer> bin = new HashMap<Double, Integer>();
		for (double d : data) {
			if (bin.containsKey(d)) {
				bin.put(d, bin.get(d) + 1);
			} else {
				bin.put(d, 1);
			}
		}
		
		double [][] hist = new double [bin.keySet().size()][2];
		ArrayList<Double> skeys = new ArrayList<Double>(bin.keySet());
		Collections.sort(skeys);
		for (int i = 0; i < hist.length; ++i) {
			hist[i][0] = skeys.get(i);
			hist[i][1] = bin.get(hist[i][0]);
		}
		return hist;
	}
	/**
	 * @param args
	 */
	public static void main(String[] args) throws Exception {
		BufferedReader in = new BufferedReader(new InputStreamReader(System.in));
		ArrayList<Double> vals = new ArrayList<Double>();
		String l;
		while ((l = in.readLine()) != null) {
			vals.add(new Double(l));
		}
		double [] data = new double [vals.size()];
		for (int i = 0; i < data.length; ++i)
			data[i] = vals.get(i);
		double [][] hist = Histogram.histogram(data);
		for (double [] d : hist)
			System.out.println(d[0] + "\t" + d[1]);
	}

}
