package util;

/**
 * This class provides basic vector calculus used in the whole project
 * @author koried@icsi.berkeley.edu
 */
public abstract class VA {
	/**
	 * c = a + b
	 */
	public static double [] add1(double [] a, double [] b) {
		double [] res = new double [a.length];
		for (int i = 0; i < a.length; ++i)
			res[i] = a[i] + b[i];
		return res;
	}
	
	/**
	 * a += b
	 */
	public static void add2(double [] a, double [] b) {
		for (int i = 0; i < a.length; ++i)
			a[i] += b[i];
	}
	
	/**
	 * c = a + b where b is a scalar
	 */
	public static double [] add3(double [] a, double b) {
		double [] res = new double [a.length];
		for (int i = 0; i < a.length; ++i)
			res[i] = a[i] + b;
		return res;
	}
	
	/**
	 * a += b where b is a scalar
	 */
	public static void add4(double [] a, double b) {
		for (int i = 0; i < a.length; ++i)
			a[i] += b;
	}
	
	/**
	 * c = a - b
	 */
	public static double [] sub1(double [] a, double [] b) {
		double [] res = new double [a.length];
		for (int i = 0; i < a.length; ++i)
			res[i] = a[i] - b[i];
		return res;
	}
	
	/**
	 * a -= b
	 */
	public static void sub2(double [] a, double [] b) {
		for (int i = 0; i < a.length; ++i)
			a[i] -= b[i];
	}
	
	/**
	 * c = a - b where b is a scalar
	 */
	public static double [] sub3(double [] a, double b) {
		return add3(a, -b);
	}
	/**
	 * a -= b where b is a scalar
	 */
	public static void sub4(double [] a, double b) {
		add4(a, -b);
	}
	
	/**
	 * c = dot-product(a, b)
	 */
	public static double mul1(double [] a, double [] b) {
		double res = 0.;
		for (int i = 0; i < a.length; ++i)
			res += a[i]*b[i];
		return res;
	}
	
	/**
	 * component wise multiplication c = a .* b
	 */
	public static double [] mul2(double [] a, double [] b) {
		double [] res = new double [a.length];
		for (int i = 0; i < a.length; ++i)
			res[i] = a[i] * b[i];
		return res;
	}
	
	/**
	 * C = a^T a
	 */
	public static double [][] mul3(double [] a, double [] b) {
		double [][] res = new double [a.length][b.length];
		for (int i = 0; i < a.length; ++i)
			for (int j = 0; j < b.length; ++j)
				res[i][j] = a[i] * b[j];
		return res;
	}
	
	/**
	 * c = a * b where b is a scalar
	 */
	public static double [] mul4(double [] a, double b) {
		double [] res = new double [a.length];
		for (int i = 0; i < a.length; ++i)
			res[i] = a[i] * b;
		return res;
	}
	
	/**
	 * a *= b where b is a scalar
	 */
	public static void mul5(double [] a, double b) {
		for (int i = 0; i < a.length; ++i)
			a[i] *= b;
	}
	
	/**
	 * a *= b component wise
	 */
	public static void mul6(double [] a, double [] b) {
		for (int i = 0; i < a.length; ++i)
			a[i] *= b[i];
	}
	
	/**
	 * component wise division c = a ./ b
	 */
	public static double [] div1(double [] a, double [] b) {
		double [] res = new double [a.length];
		for (int i = 0; i < a.length; ++i)
			res[i] = a[i] / b[i];
		return res;
	}
	
	/**
	 * c = a / b where b is a scalar
	 */
	public static double [] div2(double [] a, double b) {
		double [] res = new double [a.length];
		for (int i = 0; i < a.length; ++i)
			res[i] = a[i] / b;
		return res;
	}
	
	/**
	 * a /= b where b is a scalar
	 */
	public static void div3(double [] a, double b) {
		for (int i = 0; i < a.length; ++i)
			a[i] /= b;
	}
	
	/**
	 * a /= b component wise
	 */
	public static void div4(double [] a, double [] b) {
		for (int i = 0; i < a.length; ++i)
			a[i] /= b[i];
	}
	
	/**
	 * |v|
	 */
	public static double norm1(double [] v) {
		double r = 0;
		for (double d : v) 
			r += Math.abs(d);
		return r;
	}
	
	/**
	 * ||v||^2
	 */
	public static double norm2(double [] v) {
		double r = 0;
		for (double d : v)
			r += d*d;
		return Math.sqrt(r);
	}
}
