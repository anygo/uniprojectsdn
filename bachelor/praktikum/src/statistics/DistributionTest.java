package statistics;

import java.util.*;

/**
 * A collection of statistical tests for normal distribution of data
 * 
 * @author sikoried
 *
 */
public abstract class DistributionTest {
	/**
	 * Test whether or not a given data array satisfies a normal distribution
	 * using the Anderson Darling test at the given significance niveau.
	 * @param data sample data, will be sorted afterwards
	 * @param p significance niveau (0.1, 0.05, 0.025, 0.001, 0.0001)
	 * @return true if data seems to be normal distributed
	 */
	public static boolean andersonDarlingNormal(double [] data, double p) {
		// available critical values, feel free to extend
		HashMap<Double, Double> adt_map = new HashMap<Double, Double>();
		adt_map.put(0.1000, 0.656);
		adt_map.put(0.0500, 0.787);
		adt_map.put(0.0250, 0.918);
		adt_map.put(0.0010, 1.092);
		adt_map.put(0.0001, 1.8692);
		
		return andersonDarlingNormalCriticalValue(data) < adt_map.get(p);
	}
	
	/**
	 * Compute the Anderson Darling critical value for the given 1D data and 
	 * significance niveau. The larger the value, the more likely the data is
	 * result of a normal distribution
	 * @param data
	 * @return
	 */
	public static double andersonDarlingNormalCriticalValue(double [] data) {
		// normalize to N(0,1)
		double m = 0;
		double s = 0;
		int n = data.length;
		for (int i = 0; i < n; ++i) {
			m += data[i];
			s += data[i] * data[i];
		}
		m /= n;
		s = Math.sqrt(s / n - m * m);

		double [] data_mod = new double [data.length];
		for (int i = 0; i < n; ++i)
			data_mod[i] = (data[i] - m) / s;
		
		// sort
		Arrays.sort(data_mod);
		
		// compute F values
		for (int i = 0; i < n; ++i)
			data_mod[i] = cnf_N01(data_mod[i]);
		
		// evaluate
		return andersonDarlingSatistic(data_mod, true);
	}
	
	/**
	 * Hart, J.F. et al, 'Computer Approximations', Wiley 1968 (FORTRAN transcript)
	 */
	private static double cnf_N01(double z) {
		double zabs;
		double p;
		double expntl, pdf;

		final double p0 = 220.2068679123761;
		final double p1 = 221.2135961699311;
		final double p2 = 112.0792914978709;
		final double p3 = 33.91286607838300;
		final double p4 = 6.373962203531650;
		final double p5 = .7003830644436881;
		final double p6 = .3526249659989109E-01;

		final double q0 = 440.4137358247522;
		final double q1 = 793.8265125199484;
		final double q2 = 637.3336333788311;
		final double q3 = 296.5642487796737;
		final double q4 = 86.78073220294608;
		final double q5 = 16.06417757920695;
		final double q6 = 1.755667163182642;
		final double q7 = .8838834764831844E-1;

		final double cutoff = 7.071;
		final double root2pi = 2.506628274631001;

		zabs = Math.abs(z);
		if (z > 37.0) 
			return 1.;
		if (z < -37.0) 
			return 0.;

		expntl = StrictMath.exp(-.5*zabs*zabs);
		pdf = expntl/root2pi;

		if (zabs < cutoff) {
			p = expntl*((((((p6*zabs + p5)*zabs + p4)*zabs + p3)*zabs +
					p2)*zabs + p1)*zabs + p0)/(((((((q7*zabs + q6)*zabs +
							q5)*zabs + q4)*zabs + q3)*zabs + q2)*zabs + q1)*zabs +
							q0);
		} else {
			p = pdf/(zabs + 1.0/(zabs + 2.0/(zabs + 3.0/(zabs + 4.0/
					(zabs + 0.65)))));
		}

		if (z < 0.)
			return p;
		else
			return 1. - p;
	}

	/**
	 * Compute the Anderson Darling statistic
	 * 
	 * @param sortedData Sample data normalized to (0,1) and sorted
	 * @param normalize Perform Stephen's normalization
	 * @return
	 */
	private static double andersonDarlingSatistic(double [] F, boolean normalize) {
		int n = F.length;

		// accumulate
		double sum = 0.;
		for (int i = 0; i < n; ++i) {
			double z1 = StrictMath.log(F[i]);
			double z2 = StrictMath.log1p(-F[n-i-1]); // supposed to be more accurate
			sum += (2.*i + 1) * (z1 + z2);
		}

		double a2 = -sum / (double) n - (double) n;

		if (normalize)
			a2 *= (1. + 4./(double) n - 25./((double) (n*n)) );

		return a2;
	}
	
	public static void main(String [] args) {
		java.util.Random rand = new java.util.Random();
		double [] d = new double [10000];
		for (int i = 0; i < d.length; ++i)
			d[i] = rand.nextGaussian();
		
		System.out.println("0.1000: " + andersonDarlingNormal(d, 0.1000));
		System.out.println("0.0500: " + andersonDarlingNormal(d, 0.0500));
		System.out.println("0.0250: " + andersonDarlingNormal(d, 0.0250));
		System.out.println("0.0010: " + andersonDarlingNormal(d, 0.0010));
		System.out.println("0.0001: " + andersonDarlingNormal(d, 0.0001));
	}
}
