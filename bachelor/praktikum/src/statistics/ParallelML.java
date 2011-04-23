package statistics;

import java.io.*;
import io.*;
import java.util.*;
import java.util.concurrent.*;

/**
 * A parallel implementation of the maximum likelihood estimation. Uses a 
 * ChunkedDataSet and high level threads for a convenient and easy parallelization.
 * 
 * @author sikoried
 */
public class ParallelML {
	/** number of cpus to use */
	private int numThreads;
	
	/** feature dimension */
	private int fd;
	
	/** data set (synchronized) */
	private ChunkedDataSet data;
	
	/** restrict to diagonal covariances */
	private boolean diagonalCovariance;
	
	/** computed estimate */
	private Density estimate = null;
	
	/** number of processed samples */
	private long samples = 0;
	
	/**
	 * Generate a new parallel ML estimator.
	 * @param featureDimension
	 * @param data
	 * @param numCores
	 * @param diagonalCovariance
	 */
	public ParallelML(int featureDimension, ChunkedDataSet data, int numCores, boolean diagonalCovariance) {
		this.numThreads = numCores;
		this.data = data;
		this.fd = featureDimension;
		this.diagonalCovariance = diagonalCovariance;
	}
	
	/**
	 * Return the estimated density. Caches the result for subsequent calls.
	 */
	public Density mlEstimate() throws IOException, InterruptedException {
		System.err.println("ParallelML.estimate(): BEGIN");
		
		// already done?
		if (estimate != null)
			return estimate;
		
		CountDownLatch latch = new CountDownLatch(numThreads);
		ExecutorService e = Executors.newFixedThreadPool(numThreads);
		Estimator [] threads = new Estimator [numThreads];
		
		System.err.println("ParallelML.estimate(): starting thread pool (1)");
		
		// first part: compute means
		for (int i = 0; i < numThreads; ++i)
			e.execute(threads[i] = new Estimator(latch));
		
		// wait for all threads to finish
		latch.await();
		
		// rewind the list
		data.rewind();
		
		System.err.println("ParallelML.estimate(): normalizing mean value");

		// normalize the means, save to array, give it to the estimator
		double [] mue = new double [fd];
		for (Estimator es : threads) {
			for (double [] m : es.partial) {
				for (int i = 0; i < fd; ++i)
					mue[i] += m[i] / samples;
			}
		}
		
		System.err.println("ParallelML.estimate(): starting thread pool (2)");
		
		// second part: compute the covariances
		latch = new CountDownLatch(numThreads);
		
		for (int i = 0; i < numThreads; ++i)
			e.execute(threads[i] = new Estimator(latch, mue));
		
		// wait for all threads to finish
		latch.await();
		
		// rewind the list
		data.rewind();
		
		System.err.println("ParallelML.estimate(): normalizing covariance");
		
		// normalize the covariance
		double [] cov = diagonalCovariance ? new double [fd] : new double [fd * (fd + 1) / 2];
		for (Estimator es : threads) {
			for (double [] c : es.partial) {
				for (int i = 0; i < cov.length; ++i)
					cov[i] += c[i] / samples;
			}
		}
		
		// create the density
		estimate = (diagonalCovariance ? 
					new DensityDiagonal(1., mue, cov) : 
					new DensityFull(1., mue, cov));
		
		// update the internals
		estimate.update();
		
		// make sure the thread pool is done
		e.shutdownNow();
		
		return estimate;
	}
	
	private synchronized void processedSamples(long num) {
		samples += num;
	}
	
	/**
	 * Create a new Thread to compute the ML estimate
	 * 
	 * @author sikoried
	 *
	 */
	private class Estimator implements Runnable {

		CountDownLatch latch;
		double [] buf = new double [fd];
		
		/** for the second pass, we have a mean value for the covariance computation */
		double [] mue = null;
		
		/** current partial estimate: either mean or covariance */
		double [] est = null;
		
		/** number of processed chunks */
		long cnt_chunk = 0;
		
		/** number of processed frames */
		long cnt_frame = 0;
		
		/** container for the unnormalized values */
		LinkedList<double []> partial = new LinkedList<double []>();
		
		/**
		 * Construct an Estimator for the mean value
		 * @param latch
		 */
		Estimator(CountDownLatch latch) {
			this.latch = latch;
			est = new double [fd];
		}
		
		/**
		 * Construct an Estimator for the covariance
		 * 
		 * @param latch
		 * @param mue
		 */
		Estimator(CountDownLatch latch, double [] mue) {
			this.latch = latch;
			this.mue = mue;
			est = diagonalCovariance ? new double [fd] : new double [fd * (fd + 1) / 2];
		}
		
		public void run() {
			try {
				ChunkedDataSet.Chunk chunk;
				
				while ((chunk = data.nextChunk()) != null) {
					FrameReader source = chunk.reader;
						
					// reset the estimate for this chunk
					for (int i = 0; i < est.length; ++i)
						est[i] = 0.;
					
					while (source.read(buf)) {
						if (mue == null) {
							// mean value
							for (int i = 0; i < fd; ++i)
								est[i] += buf[i];
						} else {
							for (int i = 0; i < fd; ++i)
								buf[i] -= mue[i];
							
							if (diagonalCovariance) {
								for (int i = 0; i < fd; ++i)
									est[i] += (buf[i] - mue[i])*(buf[i] - mue[i]);
							} else {
								// lower triangular matrix in packed storage
								int k = 0;
								for (int i = 0; i < fd; ++i)
									for (int j = 0; j <= i; ++j)
										est[k++] += buf[i] * buf[j];
							}
						}
						cnt_frame++;
					}
					
					// add this partial estimate to the list
					partial.add(est.clone());
					
					cnt_chunk++;
				}
				
				// only count samples in the first pass
				if (mue == null)
					processedSamples(cnt_frame);
				
				System.err.println("ParallelML.Estimator#" + Thread.currentThread().getId() + ".run(): processed " + cnt_frame + " in " + cnt_chunk + " chunks");
				
			} catch (IOException e) {
				System.err.println("Exception in Estimator Thread #" + Thread.currentThread().getId() + ": " + e.toString());
			} finally {
				// we're done here
				latch.countDown();
			}
		}
	}
}
