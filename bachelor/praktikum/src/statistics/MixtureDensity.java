package statistics;

import java.io.*;

import util.IOUtil;

/**
 * A Gaussian mixture density using either diagonal or full covariance matrices.
 * 
 * @author sikoried
 *
 */
public class MixtureDensity implements Serializable {
	private static final long serialVersionUID = 1L;

	/** number of densities */
	public int nd;
	
	/** feature dimension */
	public int fd;
	
	/** mixture id */
	public int id;
	
	/** score after evaluation (including priors, or course) */
	public transient double score;
	
	/** log likelihood accumulator */
	public transient double llh = 0.;
	
	/** component densities */
	public Density [] components;
	
	/** give it a name if you want... */
	public String name = null;
	
	public boolean diagonal;
	
	/**
	 * Create a new MixtureDensity.
	 * @param featureDimension feature dimension
	 * @param numberOfDensities number of densities
	 * @param diagonalCovariances
	 */
	public MixtureDensity(int featureDimension, int numberOfDensities, boolean diagonalCovariances) {
		this.nd = numberOfDensities;
		this.fd = featureDimension;
		components = new Density [nd];
		diagonal = diagonalCovariances;
		
		for (int i = 0; i < nd; ++i) {
			components[i] = diagonalCovariances ? new DensityDiagonal(fd) : new DensityFull(fd);
			components[i].apr = 1. / nd;
			components[i].id = i;
		}
	}
	
	public MixtureDensity(MixtureDensity copy) {
		this.nd = copy.nd;
		this.fd = copy.fd;
		this.diagonal = copy.diagonal;
		
		components = new Density [nd];
		
		for (int i = 0; i < nd; ++i)
			components[i] = copy.components[i].clone();
	}
	
	public boolean usesDiagonalCovariances() {
		return diagonal;
	}
	
	/**
	 * Return a deep copy of this instance
	 */
	public MixtureDensity clone() {
		return new MixtureDensity(this);
	}
	
	/**
	 * Evaluate the GMM
	 * @param x feature vector
	 * @return probability of that mixture
	 */
	public double evaluate(double [] x) {
		score = 0.;
		for (Density d : components) {
			score += d.evaluate(x);
			llh += d.lh;
		}
		return score;
	}
	
	/**
	 * Return the index of the highest scoring density (without the prior or exponentiation!)
	 * @param x
	 * @return
	 */
	public int classify(double [] x, boolean withPriors) {
		// evaluate all densities
		components[0].evaluate(x);
		
		// find the maximum one 
		double max = components[0].score;
		int maxid = 0;
		for (int i = 1; i < nd; ++i) {
			double p = components[i].score;
			if (p > max) {
				max = p;
				maxid = i;
			}
		}
		return maxid;
	}
	
	/** 
	 * Normalize the component scores to posteriors (call evaluate first!)
	 * @param p container to save the posteriors to
	 */
	public void posteriors(double [] p) {
		for (int i = 0; i < nd; ++i)
			p[i] = components[i].ascore / score;
	}
	
	/**
	 * Set all the elements of the components to zero
	 */
	public void clear() {
		llh = 0.;
		for (Density d : components)
			d.clear();
	}
	
	/**
	 * Generate a super vector for GMM-SVM use. The generated vector contains 
	 * (in that order) all priors, mean values and variances (if requested).
	 * @param priors include prior probabilities
	 * @param means include mean vectors
	 * @param variances include variances (diagonal covariance)
	 * @return super vector [apr1 apr2 ... mue1 mue2 ... cov1 cov2 ...]
	 */
	public double [] superVector(boolean priors, boolean means, boolean variances) {
		// determine the size
		int size = 0;
		if (priors)
			size += 1;
		if (means)
			size += fd;
		if (variances)
			size += fd;
		
		double [] sv = new double [size * nd];
		
		// copy values
		int i = 0;
		for (Density d : components)
			System.arraycopy(d.superVector(priors, means, variances), 0, sv, size * (i++), size);
		
		return sv;
	}
	
	/**
	 * Return a String representation of the mixture
	 */
	public String toString() {
		StringBuffer sb = new StringBuffer();
		sb.append("fd = " + fd + " nd = " + nd + " diagonal: " + usesDiagonalCovariances() + "\n");
		for (int i = 0; i < nd; ++i)
			sb.append(components[i].toString() + "\n");
		return sb.toString();
	}
	
	public void writeToFile(String file) throws IOException {
		ObjectOutputStream oos = new ObjectOutputStream(file == null ? System.out : new FileOutputStream(file));
		oos.writeObject(this);
	}
	
	/**
	 * fills a MixtureDensity by reading data from 2 ASCII Files. The aPriori values are set to 0.0
	 * @param meanFile : ASCII File containing the mean values of all densities.
	 * @param covaFile : ASCII File containing the cova values of all densities.
	 * @throws IOException 
	 * @throws IOException
	 */
	public static MixtureDensity fillMDFromASCII(int fd, int nd, String meanFile, String covaFile) throws IOException {	
		
		MixtureDensity md = new MixtureDensity(fd,nd,true);
		
		double[] mean = new double[nd * fd];
		double[] cova = new double[nd * fd];
		
		IOUtil.readFloatsFromAsciiFile(meanFile, mean);
		IOUtil.readFloatsFromAsciiFile(covaFile, cova);
		
		for (int i = 0; i < nd; ++i) {
			md.components[i].apr = 0.0;
			System.arraycopy(mean, i*fd,md.components[i].mue,0, fd);
			System.arraycopy(cova, i*fd,md.components[i].cov,0, fd);
		}           
		return md;
	}
	
	public static MixtureDensity readFromFile(String file) 
		throws IOException, ClassNotFoundException {
		
		FileInputStream fis = new FileInputStream(file);
		ObjectInputStream ois = new ObjectInputStream(fis);
		MixtureDensity md = (MixtureDensity) ois.readObject();
		
		fis.close();
		return md;
	}
	
	public static final String SYNOPSIS = 
		"usage: statistics.MixtureDensity mue,cov [mue,cov ...] > mixture\n" +
		"\n" +
		"Create a mixture with equal priors and the given densities. mue and \n" +
		"cov are comma separated lists of double values. DIAGONAL COVARIANCE ONLY!\n";
	
	public static void main(String [] args) throws Exception {
		if (args.length < 1) {
			System.err.println(SYNOPSIS);
			System.exit(1);
		}
		
		Density [] ds = new Density [args.length];
		
		for (int i = 0; i < args.length; ++i) {
			ds[i] = DensityDiagonal.fromString(args[i]);
			ds[i].apr = 1./args.length;
			ds[i].update();
		}
		
		MixtureDensity md = new MixtureDensity(ds[0].fd, ds.length, true);
		md.components = ds;
		
		md.writeToFile(null);
	}
}
