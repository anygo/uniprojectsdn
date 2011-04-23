package bin;

import io.ChunkedDataSet;
import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.LinkedList;
import java.util.List;

import statistics.*;

public class Map {
	public static final String SYNOPSIS = 
		"MAP adaption for mixture densities, bocklet & sikoried 07/2009\n\n" +
		"Adapt an initial mixture density using the given feature data. If\n" +
		"num-iterations is specified, the MAP step is repeated.\n\n" +
		"usage: java bin.Map -i <initial> -o <adapted> [-a adaptation-mode] [-r <relevance>] [-n num-iterations] [-l list] [-f file]\n\n" +
		"adaptation-mode:\n" +
		"  'p' : update priors\n" +
		"  'm' : update priors\n" +
		"  'c' : update covariances\n\n" +
		"Default parameters: -a pmc -n 1 -r 16\n";
	
	public static void main(String[] args) throws IOException, ClassNotFoundException {
		if (args.length < 3) {
			System.err.println(SYNOPSIS);
			System.exit(1);
		}
		
		int i = 0;
		
		MixtureDensity initial = null;
		String outfile = null;
		String infile = null;
		String mode = "pmc";
		int numiters = 1;
		double r = 16.; 
		
		LinkedList<String> dataFiles = new LinkedList<String>();
		
		for (; i < args.length; ++i) {
			if (args[i].equals("-i"))
				infile = args[++i];
			else if (args[i].equals("-o"))
				outfile = args[++i];
			else if (args[i].equals("-a"))
				mode = args[++i].toLowerCase();
			else if (args[i].equals("-r"))
				r = Double.parseDouble(args[++i]);
			else if (args[i].equals("-l")) {
				BufferedReader br = new BufferedReader(new FileReader(args[++i]));
				String buf;
				while ((buf = br.readLine()) != null)
					dataFiles.add(buf);
				br.close();
			} else if (args[i].equals("-f")) {
				dataFiles.add(args[++i]);
			} else
				System.err.println("ignoring unknown option " + args[i]);
		}
		
		if (infile == null) {
			System.err.println("Map.main(): no input file specified");
			System.exit(1);
		}
		
		if (outfile == null) {
			System.err.println("Map.main(): no output file specified");
			System.exit(1);
		}
		
		// any data?
		if (dataFiles.size() == 0) {
			System.err.println("Map.main(): No data files provided! Exitting...");
			System.exit(1);
		}
		
		// read initial density
		System.err.println("Map.main(): reading initial model...");
		initial = MixtureDensity.readFromFile(infile);
				
		// read all the data into the memory
		System.err.println("Map.main(): Reading feature data...");
		ChunkedDataSet ds = new ChunkedDataSet(dataFiles);
		List<Sample> data = ds.cachedData();
		System.err.println("Map.main(): " + data.size() + " samples read");
		
		System.err.println("Map.main(): performing " + numiters + " MAP steps (r = " + r + ", mode=" + mode + ")");
		MixtureDensity adapted = Trainer.map(initial, data, r, numiters, mode);
		
		System.err.println("Map.main(): writing adapted mixture to " + outfile + "...");
		adapted.writeToFile(outfile);
	}
}
