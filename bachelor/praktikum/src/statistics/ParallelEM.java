package statistics;

import io.*;
import java.io.*;
import java.util.concurrent.*;

import util.VA;

/**
 * A parallel implementation of the EM algorithm. Uses an initial mixture and
 * a ChunkedDataSet to update the parameters.
 * 
 * @author sikoried
 *
 */
public final class ParallelEM {
	/** number of threads (= CPUs) to use */
	private int numThreads = 0;
	
	/** data set to use; do not forget to rewind if required! */
	private ChunkedDataSet data = null;
	
	/** previous estimate */
	public MixtureDensity previous = null;
	
	/** current estimate */
	public MixtureDensity current = null;
	
	/** number of components */
	private int nd;
	
	/** feature dimension */
	private int fd;
	
	/** diagonal covariances? */
	private boolean dc;
	
	/** number of iterations performed by this instance */
	public int ni = 0;
	
	/**
	 * Generate a new Estimator for parallel EM iterations.
	 * 
	 * @param initial Initial mixture to start from (DATA IS MODIFIED)
	 * @param data data set to use
	 * @param numThreads number of threads (= CPUs)
	 * @throws IOException
	 */
	public ParallelEM(MixtureDensity initial, ChunkedDataSet data, int numThreads) 
		throws IOException {
		this.data = data;
		this.numThreads = numThreads;
		this.current = initial;
		this.fd = initial.fd;
		this.nd = initial.nd;
		this.dc = initial.usesDiagonalCovariances();
	}
	
	/**
	 * Set the data set to work on
	 */
	public void setChunkedDataSet(ChunkedDataSet data) {
		this.data = data;
	}
	
	/**
	 * Set the number of threads for the next iteration
	 */
	public void setNumberOfThreads(int num) {
		numThreads = num;
	}
	
	/**
	 * Perform a number of EM iterations
	 * @param iterations
	 * @throws IOException
	 * @throws InterruptedException
	 */
	public void iterate(int iterations) throws IOException, InterruptedException {
		while (iterations-- > 0)
			iterate();
	}
	
	/**
	 * Perform one EM iteration
	 * @throws IOException
	 * @throws InterruptedException
	 */
	public void iterate() throws IOException, InterruptedException {
		System.err.println("ParallelEM.iterate(): BEGIN iteration " + (++ni));
		
		// each thread gets a partial estimate and a working copy of the current
		MixtureDensity [] partialEstimates = new MixtureDensity[numThreads];
		MixtureDensity [] workingCopies = new MixtureDensity[numThreads];
		
		// save the old mixture, put zeros in the current
		previous = current.clone();
		
		ExecutorService e = Executors.newFixedThreadPool(numThreads);
		
		// BEGIN EM PART1: accumulate the statistics
		CountDownLatch latch = new CountDownLatch(numThreads);
		
		for (int i = 0; i < numThreads; ++i)
			e.execute(new Worker(workingCopies[i] = current.clone(), partialEstimates[i] = new MixtureDensity(fd, nd, dc), latch));
		
		// wait for all jobs to be done
		latch.await();
		
		// make sure the thread pool is done
		e.shutdownNow();
		
		// rewind the list 
		data.rewind();
		
		// BEGIN EM PART2: combine the partial estimates
		current.clear();
		
		// sum of all posteriors
		double ps = 0.;
		
		for (MixtureDensity est : partialEstimates) {
			for (int i = 0; i < nd; ++i) {
				Density source = est.components[i];
				Density target = current.components[i];
				
				target.apr += source.apr;
				ps += source.apr;				
				
				for (int j = 0; j < fd; ++j)
					target.mue[j] += source.mue[j];
				
				for (int j = 0; j < target.cov.length; ++j)
					target.cov[j] += source.cov[j];
			}
		}
		
		// normalize means and covariances
		for (int i = 0; i < nd; ++i) {
			Density d = current.components[i];
			
			double [] mue = d.mue;
			double [] cov = d.cov;
			
			VA.div3(mue, d.apr);
			VA.div3(cov, d.apr);
			
			d.apr /= ps;
			
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
			
			// update the internals of the new estimate
			d.update();
		}
		
		// END EM Part2

		System.err.println("ParallelEM.iterate(): END");
	}
	
	/**
	 * First part of the EM: Accumulate posteriors, prepare priors and mean
	 */
	private class Worker implements Runnable {
		MixtureDensity work, est;
		CountDownLatch latch;
		
		/** feature buffer */
		double [] f;
		
		/** posterior buffer */
		double [] p;
		
		/** number of chunks processed by this thread */
		int cnt_chunk = 0;
		
		/** number of frames processed by this thread */
		int cnt_frame = 0;
		
		Worker(MixtureDensity workingCopy, MixtureDensity partialEstimate, CountDownLatch latch) {
			this.latch = latch;
			this.work = workingCopy;
			this.est = partialEstimate;
			
			// init the buffers
			f = new double [fd];
			p = new double [nd];
			
			// make sure the estimate is cleared up!
			est.clear();
		}
		
		/**
		 * Main thread routine: read as there are chunks, compute posteriors,
		 * update the accus
		 */
		public void run() {
			try {
				ChunkedDataSet.Chunk chunk;
				
				// as long as we have chunks to do... NB: data is (synchronized) from ParallelEM!
				while ((chunk = data.nextChunk()) != null) {
					FrameReader source = chunk.reader;
						
					while (source.read(f)) {
						work.evaluate(f);
						work.posteriors(p);
												
						for (int i = 0; i < nd; ++i) {
							// prior
							est.components[i].apr += p[i];
							
							double [] mue = est.components[i].mue;
							double [] cov = est.components[i].cov;
							
							// mean, only if full covariances (see below!)
							if (!dc) {
								for (int j = 0; j < fd; ++j)
									mue[j] += p[i] * f[j];
							}
							
							// covariance
							if (dc) {
								for (int j = 0; j < fd; ++j) {
									mue[j] += p[i] * f[j];
									cov[j] += p[i] * f[j] * f[j];
								}
							} else {
								int l = 0;
								for (int j = 0; j < fd; ++j)
									for (int k = 0; k <= j; ++k)
										cov[l++] += p[i] * f[j] * f[k];
							}
							
						}
						
						cnt_frame++;
					}
					
					cnt_chunk++;
				}
				
				System.err.println("ParallelEM.Worker#" + Thread.currentThread().getId() + ".run(): processed " + cnt_frame + " in " + cnt_chunk + " chunks");
			
			} catch (IOException e) {
				System.err.println("ParallelEM.Worker#" + Thread.currentThread().getId() + ".run(): IOException: " + e.toString());
			} finally {
				// notify the main thread
				latch.countDown();
			}
		}
	}
}
