package statistics;

import java.util.List;
import util.VA;
import java.util.Iterator;

/**
 * Implementation of sequential training algorithms for (mixture) densities. 
 * These are: EM, MAP, ML. 
 * 
 * @author bocklet,sikoried
 *
 */
public abstract class Trainer {
	/**
	 * Standard (single-core) maximum likelihood estimation for a single 
	 * Gaussian density
	 * 
	 * @param data
	 * @param diagonalCovariances 
	 * @return
	 * @throws TrainingException
	 */
	public static Density ml(List<Sample> data, boolean diagonalCovariances) {
		int fd = data.get(0).x.length;
		double scale = 1. / data.size();

		Density d = (diagonalCovariances ? new DensityDiagonal(fd) : new DensityFull(fd));
		
		// accumulate...
		for (Sample s : data) {
			for (int i = 0; i < fd; ++i) {
				d.mue[i] += s.x[i] * scale;
				
				if (diagonalCovariances)
					d.cov[i] += s.x[i] * s.x[i] * scale;
				else {
					int l = 0;
					for (int j = 0; j < fd; ++j)
						for (int k = 0; k <= j; ++k)
							d.cov[l++] += s.x[j] * s.x[k] * scale;
				}
			}
		}
		
		// normalize...
		if (diagonalCovariances) {
			for (int i = 0; i < fd; ++i)
				d.cov[i] -= (d.mue[i] * d.mue[i]);
		} else {
			int k = 0;
			for (int i = 0; i < fd; ++i)
				for (int j = 0; j <= i; ++j)
					d.cov[k++] -= (d.mue[i] * d.mue[j]);
		}
		
		// update internal stats 
		d.update();

		return d;
	}
	
	/**
	 * Perform a number of EM iterations (single-core, cached posteriors) using
	 * the initial density and the given data.
	 * 
	 * @param initial
	 * @param data
	 * @param iterations
	 * @return
	 */
	public static MixtureDensity em(MixtureDensity initial, List<Sample> data, int iterations) {
		MixtureDensity estimate = initial;
		while (iterations-- >= 0)
			estimate = em(estimate, data);
		return estimate;
	}
	
	/**
	 * Given an initial mixture density, perform a single EM iteration w/ out
	 * the use of parallelization. <br />
	 * Note that this version makes use of the following reformulation:
	 * K = (sum_i gamma_i (x - mue)(x - mue)^T) / (sum_i gamma_i) = <br />
	 *   = ((sum_i gamma_i xx^T) / (sum_i gamma_i)) - mue mue^T
	 * 
	 * @param initial
	 * @param data
	 * @return
	 * @throws TrainingException
	 */
	public static MixtureDensity em(MixtureDensity initial, List<Sample> data) {
		boolean dc = initial.usesDiagonalCovariances();

		int nd = initial.nd;
		int fd = initial.fd;
	
		// will hold the posteriors
		double [] p = new double [nd];
		
		// accumulated posteriors
		double asum = 0.;
		double [] sp = new double[nd];
		
		// first, accumulate the statistics
		System.err.println("Trainer.em(): computing statistics...");
		
		MixtureDensity em_estim2 = new MixtureDensity(fd, nd, dc);
		Density[] d2 = em_estim2.components;
		
		Iterator<Sample> xiter = data.iterator();
		while (xiter.hasNext()) {
			double[] x = xiter.next().x;
			
			// evaluate and get posteriors
			initial.evaluate(x);
			initial.posteriors(p);
			
			// for all densities
			for (int j = 0; j < nd; ++j) {
				// priors
				sp[j] += p[j];
				asum += p[j];
				
				double [] mue = d2[j].mue;
				double [] cov = d2[j].cov;
			
				// mean values, only if full cov (see below!)
				if (!dc) {
					for (int k = 0; k < fd; ++k)
						mue[k] += p[j] * x[k];
				}
		
				// covariance
				if (dc) {
					for (int k = 0; k < fd; ++k) {
						mue[k] += p[j] * x[k];
						cov[k] += p[j] * x[k] * x[k];
					}
				} else {
					int m = 0;
					for (int k = 0; k < fd; ++k)
						for (int l = 0; l <= k; ++l)
							cov[m++] += p[j] * x[k] * x[l];
				}
			}			
		}

		// normalize priors, means and covariance
		System.err.println("Trainer.em(): normalizing...");
		
		for (int i = 0; i < nd; ++i) {
			d2[i].apr = sp[i] / asum;
			
			double [] mue = d2[i].mue;
			double [] cov = d2[i].cov;
			
			VA.div3(mue, sp[i]);
			VA.div3(cov, sp[i]);
			
			// conclude covariance computation
			if (dc) {
				for (int j = 0; j < fd; ++j)
					cov[j] -= mue[j] * mue[j];
			} else {
				int l = 0;
				for (int j = 0; j < fd; ++j)
					for (int k = 0; k <= j; ++k)
						cov[l++] -= mue[j] * mue[k];
			}
		}

		// update the internals
		for (Density dens : d2)
			dens.update();

		return em_estim2;
	}
	
	/**
	 * Perform a number of MAP iterations, based on the initial estimate
	 * 
	 * @param initial
	 * @param data
	 * @param iterations
	 * @param update String indicating which parameters to update: p for prior, m for mean, c for covariance; if null, "pmc" (i.e. all parameters) is assumed
	 * @return
	 */
	public static MixtureDensity map(MixtureDensity initial, List<Sample> data, double r, int iterations, String update) {
		MixtureDensity estimate = initial;
		while (iterations-- > 0)
			estimate = map(estimate, data, r, update);
		return estimate;
	}
	
	/**
	 * Perform a single MAP iteration on the given initial estimate
	 * 
	 * @param initial
	 * @param r Relevance factor
	 * @param update String indicating which parameters to update: p for prior, m for mean, c for covariance; if null, "pmc" (i.e. all parameters) is assumed
	 * @return
	 */
	public static MixtureDensity map(MixtureDensity initial, List<Sample> data, double r, String update) {
		
		boolean updatePriors = true;
		boolean updateMeans = true;
		boolean updateCovariance = true;
		
		if (update != null) {
			update = update.toLowerCase();
			if (update.indexOf("p") < 0)
				updatePriors = false;
			if (update.indexOf("m") < 0)
				updateMeans = false;
			if (update.indexOf("c") < 0)
				updateCovariance = false;
		}
		
		if (!(updatePriors || updateMeans || updateCovariance)) {
			System.err.println("Trainer.Map(): no parameter update selected, returning initial estimate");
			return initial;
		}
		
		int n = data.size();
		int nd = initial.nd;
		int fd = initial.fd;
		
		boolean diagonal = initial.usesDiagonalCovariances();
	
		// accumulate posteriors
		double [] sp = new double[nd];
		double [] p = new double [nd];
		
		// new container
		MixtureDensity adapted = new MixtureDensity(fd, nd, diagonal);
		
		Density[] d1 = initial.components;
		Density[] d2 = adapted.components;
		
		// accumulate the statistics
		Iterator<Sample> xiter = data.iterator();
		for (int i = 0; i < n; ++i) {
			double[] x = xiter.next().x;
			
			initial.evaluate(x);
			initial.posteriors(p);
			
			for (int j = 0; j < nd; ++j) {
				// priors
				sp[j] += p[j];
			
				double [] mue = d2[j].mue;
				double [] cov = d2[j].cov;
				
				// means
				for (int k = 0; k < fd; ++k)
					mue[k] += p[j] * x[k];
			
				// covariance
				if (diagonal) {
					for (int k = 0; k < fd; ++k)
						cov[k] += p[j] * x[k] * x[k];
				} else {
					int m = 0;
					for (int k = 0; k < fd; ++k)
						for (int l = 0; l <= k; ++l)
							cov[m++] += p[j] * x[k] * x[l];
				}
			}
		}
		
		// normalize the statistics
		for (int i = 0; i < nd; ++i) {
			VA.div3(d2[i].mue, sp[i]);
			VA.div3(d2[i].cov, sp[i]);
		}
		
		// now combine the two estimates using the relevance factor
		double priorSum = 0.;
		for (int i = 0; i < nd; ++i){
			double alpha = sp[i] / (r + sp[i]);
			
			// update prior
			if (updatePriors) {
				d2[i].apr = ((alpha * sp[i]) / (double) n) + ((1. - alpha) * d1[i].apr);
				priorSum += d2[i].apr;
			} else
				d2[i].apr = d1[i].apr;
			
			// update mean
			if (updateMeans) {
				for (int j = 0; j < fd; ++j)
					d2[i].mue[j] = (alpha * d2[i].mue[j]) + ((1. - alpha) * d1[i].mue[j]);  
			} else
				System.arraycopy(d1[i].mue, 0, d2[i].mue, 0, fd);
			
			if (updateCovariance) {
				// update covariance matrix
				for (int j = 0; j < fd; ++j){
					if(diagonal) {
						d2[i].cov[j] = 
							(alpha * d2[i].cov[j]) + 
							((1. - alpha) * (d1[i].cov[j] + (d1[i].mue[j] * d1[i].mue[j])) - 
							(d2[i].mue[j] * d2[i].mue[j]));
					} else {
						int m = 0;
						for (int k = 0; k <= j; ++k){
							d2[i].cov[m] = 
								(alpha * d2[i].cov[m]) + 
								((1. - alpha) * (d1[i].cov[m] + (d1[i].mue[j] * d1[i].mue[k])) - 
								(d2[i].mue[j] * d2[i].mue[k]));
							m++;
						}
					}
				}
			} else
				System.arraycopy(d1[i].cov, 0, d2[i].cov, 0, d1[i].cov.length);
		}
		
		// normalize priors
		if (updatePriors) {
			for (int i = 0; i < nd; ++i)
				d2[i].apr /= priorSum;
		}
		
		// update the internals
		for (Density d : adapted.components)
			d.update();
		
		return adapted;
	}	
}
