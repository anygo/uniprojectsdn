package bin;

import exceptions.*;
import statistics.*;
import statistics.Initialization.DensityRankingMethod;
import io.*;
import java.util.*;
import java.io.*;

public class Initializer {

	public static final String SYNOPSIS = 
		"Mixture initializer, sikoried 06/2009\n" +
		"usage: java bin.Initializer strategy num-clusters {diagonal: true|false} outfile [-l list] [feature-file1]\n" +
		"\n" +
		"Available strategies:\n" +
		"  knn\n" +
		"    Find the clusters by iteratively distribute the data into the\n" +
		"    num-cluster clusters, refining the centroid in each step.\n\n" +
		"  g [strategy]\n" +
		"    Hierarchical, statistically driven Gaussian clustering, similar\n" +
		"    to the LBG algorithm.\n" +
		"    Available strategies:\n" +
		"      none     : split cluster if not normally distributed (no re-ranking)\n" +
		"      cov      : split the cluster with highest covariance\n" +
		"      sum_ev   : split the cluster with the highest sum of eigen values of\n" +
		"                 the covariance\n" +
		"      diff_ev  : split the cluster with the highest difference in eigen\n" +
		"                 values\n" +
		"      ad_score : split the cluster with the highest Anderson-Darling\n" +
		"                 statistics\n" +
		"      ev       : compare densities by the largest EV";
	
	public static void main(String[] args) throws IOException, TrainingException {
		// check arguments
		if (args.length < 4) {
			System.err.println(SYNOPSIS);
			System.exit(1);
		}
		
		// read arguments
		int i = 0;
		String strategy = args[i++];
		String strategy_arg = null;
		if (strategy.equals("g"))
			strategy_arg = args[i++];
		int numc = Integer.parseInt(args[i++]);
		boolean diag = Boolean.parseBoolean(args[i++]);
		String outfile = args[i++];
		
		// read the data file names		
		System.err.println("Initializer.main(): Reading feature data...");
		LinkedList<String> dataFiles = new LinkedList<String>();
		for (; i < args.length; ++i) {
			if (args[i].equals("-l")) {
				System.err.println("Initializer.main(): processing list file '" + args[i+1] + "'");
				// read list file
				BufferedReader br = new BufferedReader(new FileReader(args[++i]));
				String buf;
				while ((buf = br.readLine()) != null)
					dataFiles.add(buf);
				br.close();
			} else
				dataFiles.add(args[i]);
		}
		
		if (dataFiles.size() == 0) {
			System.err.println("Initializer.main(): No data files provided! Exitting...");
			System.exit(1);
		}
		
		// read all the data into the memory
		ChunkedDataSet ds = new ChunkedDataSet(dataFiles);
		List<Sample> data = ds.cachedData();
		
		System.err.println("Initializer.main(): " + data.size() + " samples read");
		
		MixtureDensity estimate = null;
		
		System.err.println("Initializer.main(): Starting clustering...");
		
		if (strategy.equals("knn"))
			estimate = Initialization.kMeansClustering(data, numc, diag);
		else if (strategy.equals("g")) {
			if (strategy_arg.equals("none"))
				estimate = Initialization.gMeansClustering(data, 0.1, numc, diag);
			else if (strategy_arg.equals("cov"))
				estimate = Initialization.hierarchicalGaussianClustering(data, numc, diag, DensityRankingMethod.COVARIANCE);
			else if (strategy_arg.equals("sum_ev"))
				estimate = Initialization.hierarchicalGaussianClustering(data, numc, diag, DensityRankingMethod.SUM_EIGENVALUE);
			else if (strategy_arg.equals("diff_ev"))
				estimate = Initialization.hierarchicalGaussianClustering(data, numc, diag, DensityRankingMethod.EV_DIFFERENCE);
			else if (strategy_arg.equals("ad_score"))
				estimate = Initialization.hierarchicalGaussianClustering(data, numc, diag, DensityRankingMethod.AD_STATISTIC);
			else if (strategy_arg.equals("ev"))
				estimate = Initialization.hierarchicalGaussianClustering(data, numc, diag, DensityRankingMethod.EV);
		} 
		else {
			System.err.println("Initializer.main(): unknown strategy '" + strategy + "'");
			System.exit(1);
		}
		
		// write the estimated parameters
		System.err.println("Initializer.main(): Writing parameters to " + outfile);
		estimate.writeToFile(outfile);
		
	}

}
