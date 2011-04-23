package statistics;

public final class DensityDiagonal extends Density {
	private static final long serialVersionUID = 1L;

	/**
	 * Create a new Density with diagonal covariance.
	 * @param dim Feature dimension
	 */
	public DensityDiagonal(int dim) {
		super(dim);
		cov = new double [fd];
	}
	
	public DensityDiagonal(DensityDiagonal copy) {
		this(copy.apr, copy.mue, copy.cov);
		this.id = copy.id;
	}
	
	public DensityDiagonal(DensityFull copy) {
		this(copy.mue.length);
		this.apr = copy.apr;
		this.id = copy.id;
		System.arraycopy(copy.mue, 0, mue, 0, fd);
		int k = 0;
		for (int i = 0; i < fd; ++i) {
			for (int j = 0; j <= i; ++j) {
				if (i == j)
					cov[i] = copy.cov[k];
				k++;
			}
		}
		
		update();
	}

	/**
	 * Create a new Density with diagonal covariance
	 * @param apr prior probability
	 * @param mue mean vector
	 * @param cov covariance vector
	 */
	public DensityDiagonal(double apr, double [] mue, double [] cov) {
		this(mue.length);
		fill(apr, mue, cov);
	}

	/** Update the internal variables. Required after modification. */
	public void update() {
		// check for NaN!
		StringBuffer fixes = new StringBuffer();
		for (int i = 0; i < fd; ++i) {
			if (Double.isNaN(mue[i])) {
				mue[i] = 1e-5;
				fixes.append(" mue[" + i + "]");
			}
			if (Double.isNaN(cov[i])) {
				cov[i] = 1e-5;
				fixes.append(" cov[" + i + "]");
			}
		}
		
		logdet = 0.;		
		for (double c : cov)
			logdet += Math.log(c);
		lapr = Math.log(apr);
		
		if (Double.isNaN(apr) || Double.isNaN(lapr)) {
			apr = 1e-10;
			lapr = Math.log(1e-10);
			fixes.append(" apr");
		}
		
		if (fixes.length() > 0)
			System.err.println("Density#" + id + ".update(): fixed NaN at:" + fixes.toString());
	}

	/**
	 * Evaluate the density for the given sample vector x. score keeps the
	 * probability (without the prior).
	 * @param x feature vector
	 * @return prior times score
	 */
	public double evaluate(double [] x) {
		// score = exp(-.5 * ( log(det) + fd*log(2*pi) + (x-mue)^T cov^-1 (x-mue)))

		// log of determinant + log(2*pi)
		score = logdet + logpiconst;
		
		// mahalanobis dist
		double h;
		for (int i = 0; i < fd; ++i) {
			h = x[i] - mue[i];
			h *= h;
			score +=  h / cov[i];
		}
		
		score *= -.5;
		
		lh = lapr + score;
		
		score = Math.exp(score + REGULARIZER);
		
		if (Double.isNaN(score) || score < MIN_PROB)
			score = MIN_PROB;
		
		ascore = apr * score;
		
		return ascore;
	}
	
	/**
	 * Create a deep copy of this instance.
	 */
	public Density clone() {
		return new DensityDiagonal(this);
	}

	/**
	 * Obtain a string representation of the density.
	 */
	public String toString() {
		StringBuffer sb = new StringBuffer();
		sb.append("apr = " + apr + "\nmue =");
		for (double m : mue)
			sb.append(" " + m);
		sb.append("\ncov =");
		for (double c : cov)
			sb.append(" " + c);
		// sb.append("\nlogdet = " + logdet + "\nlogpiconst = " + logpiconst);
		return sb.toString();
	}
	
	private static java.util.Random gen = new java.util.Random(System.currentTimeMillis());
	
	public double [] drawSample() {
		double [] x = new double [fd];
		for (int i = 0; i < fd; ++i)
			x[i] = mue[i] + gen.nextGaussian() * cov[i];
		
		return x;
	}
	
	/**
	 * Read a parameter string
	 * @param ps comma seperated list of double values, representing mean and diagonal covariance
	 * @return
	 */
	public static DensityDiagonal fromString(String ps) {
		String [] list = ps.split(",");
		
		double [] mue = new double [list.length / 2];
		double [] cov = new double [list.length / 2];
		
		for (int i = 0; i < mue.length; ++i)
			mue[i] = Double.parseDouble(list[i]);
		for (int i = 0; i < cov.length; ++i)
			cov[i] = Double.parseDouble(list[mue.length + i]);
		
		return new DensityDiagonal(1., mue, cov);
	}
}
