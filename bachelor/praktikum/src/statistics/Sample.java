package statistics;

import java.util.List;
import java.util.ArrayList;
import java.io.Serializable;
import util.VA;

/** 
 * A more sophisticated way of storing a Sample. Includes label and data,
 * provides some utility functions.
 * 
 * @author sikoried
 *
 */
public class Sample implements Serializable {
	private static final long serialVersionUID = 1L;

	/** correct label */
	public int c;
	
	/** classified label */
	public int y;
	
	/** data vector */
	public double [] x;
	
	public Sample(Sample s) {
		x = new double [s.x.length];
		y = s.y;
		c = s.c;
		System.arraycopy(s.x, 0, x, 0, s.x.length);
	}
	
	/**
	 * Generate a new sample with the correct label c
	 * @param c correct label
	 * @param x data
	 */
	public Sample(int c, double [] x) {
		this.x = new double [x.length];
		this.c = c;
		System.arraycopy(x, 0, this.x, 0, x.length);
	}
	
	/**
	 * Generate a new (empty) sample
	 * @param c correct label
	 * @param dim feature dimension
	 */
	public Sample(int c, int dim) {
		this.c = c;
		x = new double [dim];
	}
	
	/**
	 * Return a String representation
	 */
	public String toString() {
		String val = "" + c + " " + y;
		for (double d : x)
			val += " " + d;
		return val;
	}
	
	/**
	 * Create an ArrayList of Samples from a double array; one sample per row
	 * @param data rows will be samples
	 * @return ArrayList of Samples
	 */
	public static ArrayList<Sample> unlabeledArrayListFromArray(double [][] data) {
		ArrayList<Sample> n = new ArrayList<Sample>();
		for (double [] x : data)
			n.add(new Sample(0, x));
		return n;
	}
	
	/**
	 * Remove all data from a list which is not of label id
	 * @param data data set
	 * @param id target class
	 * @return
	 */
	public static ArrayList<Sample> reduceToClass(ArrayList<Sample> data, int id) {
		ArrayList<Sample> n = new ArrayList<Sample>();
		for (Sample s : data)
			if (s.c == id)
				n.add(s);
		return n;
	}
	
	/**
	 * Remove samples of a certain class from the data set
	 * @param data data set
	 * @param id target class
	 * @return
	 */
	public static ArrayList<Sample> removeClass(ArrayList<Sample> data, int id) {
		ArrayList<Sample> n = new ArrayList<Sample>();
		for (Sample s : data)
			if (s.c != id)
				n.add(s);
		return n;
	}
	
	/**
	 * Subtract the mean value from all samples
	 * @param data
	 * @return
	 */
	public static Sample meanSubstract(List<Sample> data) {
		Sample mean = new Sample(data.get(0));
		
		for (int i = 1; i < data.size(); ++i)
			VA.add2(mean.x, data.get(i).x);
		
		VA.div3(mean.x, data.size());
		
		for (Sample s : data)
			VA.sub2(s.x, mean.x);
		
		return mean;
	}
}
