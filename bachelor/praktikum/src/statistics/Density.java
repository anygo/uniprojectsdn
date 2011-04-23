package statistics;

import java.io.Serializable;

/**
 * The abstract Density class provides the basic assets of a Gaussian density.
 * 
 * @author sikoried
 *
 */
public abstract class Density implements Serializable {
	private static final long serialVersionUID = 1L;

	/** regularizer for density score */
	public static final double REGULARIZER = 1e-10;
	
	/** minimum density score */
	public static final double MIN_PROB = 1e-50;
	
	/** feature dimension */
	public int fd;
	
	/** Density ID */
	public int id = 0;

	/** prior probability */
	public double apr = 1.;
	
	/** log(apr) */
	protected double lapr = 0.;
	
	/** log likelihood: log(apr*score) */
	public transient double lh = 0.;
	
	/** cached version of log(Det) */
	protected double logdet;
	
	/** cached version of the log(pi) constant */
	protected double logpiconst;
	
	/** cached score from the last evaluate call, no prior! */
	public transient double score;
	
	/** cached score from the last evaluate call, with prior! */
	public transient double ascore;
	
	/** mean vector */
	public double [] mue;
	
	/** covariance matrix: either diagonal, or packed lower triangle */
	public double [] cov;
	
	/**
	 * Create a new density with a certain feature dimension
	 * @param dim Feature dimesion
	 */
	public Density(int dim) {
		fd = dim;
		logpiconst = (double) fd * Math.log(2. * Math.PI);
		mue = new double [fd];
	}
	
	/**
	 * Evaluate the density for the given sample vector x. score keeps the
	 * probability (without the prior).
	 * @param x feature vector
	 * @return prior times score
	 */
	public abstract double evaluate(double [] x);
	
	/**
	 * Set the parameters of the density.
	 * @param apr prior probability
	 * @param mue mean vector
	 * @param cov covariance vector
	 */
	public void fill(double apr, double [] mue, double [] cov) {
		this.apr = apr;
		System.arraycopy(mue, 0, this.mue, 0, fd);
		System.arraycopy(cov, 0, this.cov, 0, cov.length);
		update();
	}
	
	/**
	 * Update the internally cached variables. Required after modification.
	 */
	public abstract void update();
	
	/**
	 * Reset all the components.
	 */
	public void clear() {
		apr = 0.;
		for (int i = 0; i < fd; ++i)
			mue[i] = 0.;
		for (int i = 0; i < cov.length; ++i)
			cov[i] = 0.;
		lapr = 0.;
		lh = 0.;
	}
	
	public double [] superVector(boolean prior, boolean mue, boolean cov) {
		int size = 0;
		
		// get size
		if (prior)
			size += 1;
		if (mue)
			size += fd;
		if (cov)
			size += fd;
		
		double [] sv = new double [size];
		
		// copy data
		int i = 0;
		if (prior)
			sv[i++] = apr;
		
		if (mue)
			for (double m : this.mue)
				sv[i++] = m;
		
		if (cov) {
			Density d = this;
			if (d instanceof DensityFull)
				d = new DensityDiagonal((DensityFull) d);
			for (double c : d.cov)
				sv[i++] = c;
		}
		
		return sv;
	}
	
	/**
	 * Clone the instance (deep copy)
	 */
	public abstract Density clone();
	
	/** Generate the gnuplot command (parametric) for this density, accounting 
	 * for the first 2 dimensions 
	 */
	public String covarianceAsGnuplot() {
		double [][] cov = new double [2][2];
		
		// build covariance matrix
		if (this instanceof DensityDiagonal) {
			cov[0][1] = cov[1][0] = 0.;
			cov[0][0] = this.cov[0];
			cov[1][1] = this.cov[1];
		} else {
			cov[0][0] = this.cov[0];
			cov[0][1] = cov[1][0] = this.cov[1];
			cov[1][1] = this.cov[2];
		}
		
		// get the eigen vectors and values
		Jama.EigenvalueDecomposition eigt = new Jama.Matrix(cov).eig();
		
		// width and height of the cov ellipse: sqrt of the eigenvalues
		double [] wh = eigt.getRealEigenvalues();
		
		wh[0] = Math.sqrt(wh[0]);
		wh[1] = Math.sqrt(wh[1]);
		
		double [][] eig = eigt.getV().getArray();
		
		// the eigen decomposition sorts the eigen values and vectors by
		// ascencending eigen value, thus we need to swap if the covariance
		// are in a descending order
		if (cov[0][0] > cov[1][1] && wh[0] < wh[1]) {
			double h = wh[0];
			wh[0] = wh[1];
			wh[1] = h;
			
			h = eig[0][0];
			eig[0][0] = eig[0][1];
			eig[0][1] = h;
			
			h = eig[1][0];
			eig[1][0] = eig[1][1];
			eig[1][1] = h;
		}
		
		// rotation angle from the eigenvector belonging to the first dimension
		double angle = Math.atan2(eig[0][1], eig[0][0]); 
		
		final String template1 = "X0 + DIM1 * cos(A) * cos(t) - DIM2 * sin(A) * sin(t)";
		final String template2 = "X1 + DIM1 * sin(A) * cos(t) + DIM2 * cos(A) * sin(t)";
		
		String rep1 = template1.replace("A", "" + angle);
		rep1 = rep1.replace("DIM1", "" + wh[0]);
		rep1 = rep1.replace("DIM2", "" + wh[1]);
		rep1 = rep1.replace("X0", "" + mue[0]);
		rep1 = rep1.replace("X1", "" + mue[1]);
		
		String rep2 = template2.replace("A", "" + angle);
		rep2 = rep2.replace("DIM1", "" + wh[0]);
		rep2 = rep2.replace("DIM2", "" + wh[1]);
		rep2 = rep2.replace("X0", "" + mue[0]);
		rep2 = rep2.replace("X1", "" + mue[1]);
		
		return rep1 + ", " + rep2;
	}
}
