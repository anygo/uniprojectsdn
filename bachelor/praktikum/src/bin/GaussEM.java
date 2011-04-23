package bin;

import java.io.*;
import io.*;
import statistics.*;
import java.util.LinkedList;

public class GaussEM {
	
	public static final String SYNOPSIS = 
		"Estimate Gaussian mixture densities using an initial estimate and a\n" + 
		"(large) data set.\n\n" +
		"usage: java bin.GaussEM <options>\n" +
		"  -i initial-model\n" +
		"    Initial estimate of the mixture density. See bin.Initializer for\n" +
		"    possible starts.\n" +
		"  -n iterations\n" +
		"    Number of EM iterations to compute.\n" +
		"  -o output-model\n" +
		"    File to write the final estimate to.\n" +
		"  -l listfile\n" +
		"    Use a list file to specify the files to read from.\n" +
		"  -p num\n" +
		"    Parallelize the EM algorithm on num cores (threads). Use 0 for \n" +
		"    maximum available number of cores. NB: -p 1 is different from -s as\n" +
		"    it doesn't cache the entire data set.\n" +
		"  -s\n" +
		"    Do a standard single-core EM with a complete caching of the data.\n" +
		"    This might be faster than -p for small problems with less files.\n" +
		"  --save-partial-estimates\n" +
		"    Write out the current estimate after each iteration (to output-model.*)\n" +
		"\n" +
		"default: -n 10 -p 0\n";
	
	public static void main(String[] args) throws IOException, Exception {
		if (args.length < 6) {
			System.err.println(SYNOPSIS);
			System.exit(1);
		}
		
		String inf = null;
		String ouf = null;
		String lif = null;
		
		boolean savePartialEstimates = false;
		
		// number of iterations
		int n = 10;
		
		// number of cores
		int c = Runtime.getRuntime().availableProcessors();
		
		for (int i = 0; i < args.length; ++i) {
			if (args[i].equals("-i"))
				inf = args[++i];
			else if (args[i].equals("-o"))
				ouf = args[++i];
			else if (args[i].equals("-p")) {
				int tc = Integer.parseInt(args[++i]);
				if (tc > c)
					throw new RuntimeException("too many cores requested!");
				if (tc > 0)
					c = tc;
			} else if (args[i].equals("-s"))
				c = -1;
			else if (args[i].equals("-n"))
				n = Integer.parseInt(args[++i]);
			else if (args[i].equals("-l"))
				lif = args[++i];
			else if (args[i].equals("--save-partial-estimates"))
				savePartialEstimates = true;
		}
		
		if (inf == null) {
			System.err.println("no input file specified");
			System.exit(1);
		}
		
		if (ouf == null) {
			System.err.println("no output file specified");
			System.exit(1);
		}
		
		if (lif == null) {
			System.err.println("no list file specified");
			System.exit(1);
		}
		
		System.err.println("Reading from " + inf + "...");
		MixtureDensity initial, estimate;
		initial = MixtureDensity.readFromFile(inf);
		
		if (savePartialEstimates)
			initial.writeToFile(ouf + ".0");
		
		if (c == -1) {
			System.err.print("Caching feature data...");
			LinkedList<Sample> data = new LinkedList<Sample>();
			ChunkedDataSet set = new ChunkedDataSet(lif);
			ChunkedDataSet.Chunk chunk;
			while ((chunk = set.nextChunk()) != null) {
				double [] buf = new double [chunk.reader.getFrameSize()];
				while (chunk.reader.read(buf))
					data.add(new Sample(0, buf));
			}
			System.err.println(data.size() + " samples cached");
			
			System.err.println("Starting " + n + " EM iterations: single-core, cached data");			
			
			estimate = initial;
			for (int i = 0; i < n; ++i) {
				estimate = Trainer.em(estimate, data);
				if (savePartialEstimates)
					estimate.writeToFile(ouf + "." + (i+1));
			}
		} else {
			System.err.println("Starting " + n + " EM iterations on " + c + " cores");
			ParallelEM pem = new ParallelEM(initial, new ChunkedDataSet(lif), c);
			
			for (int i = 0; i < n; ++i) {
				pem.iterate();
				if (savePartialEstimates)
					pem.current.writeToFile(ouf + "." + (i+1));
			}
			estimate = pem.current;
		}
		
		System.err.println("Saving new estimate...");
		estimate.writeToFile(ouf);
	}
}
