package bin;

import statistics.*;
import io.*;

public class SuperVector {

	public static final String SYNOPSIS = 
		"SuperVector generator, sikoried 07/2009\n\n" +
		"Generate super vectors from mixture densities and concatenate them to\n" +
		"a frame file.\n\n" +
		"usage: java bin.SuperVector <param-string> [model1 ...] > frame-file\n" +
		"  param-string:\n" +
		"    'p' : include priors\n" +
		"    'm' : include means\n" +
		"    'c' : include (diagonal) covariances\n";
	
	public static void main(String[] args) throws Exception {
		if (args.length < 1) {
			System.err.println(SYNOPSIS);
			System.exit(1);
		}
		
		boolean p = false;
		boolean m = false;
		boolean c = false;
		
		String paramString = args[0].toLowerCase();
		
		if (paramString.indexOf("p") >= 0)
			p = true;
		if (paramString.indexOf("m") >= 0)
			m = true;
		if (paramString.indexOf("c") >= 0)
			c = true;
		
		System.err.println("SuperVector.main(): prior=" + p + " mean=" + m + " cov=" + c);
		
		FrameWriter fw = null;
		for (int i = 1; i < args.length; ++i) {
			System.err.println("SuperVector.main(): writing super vector for " + args[i]);
			MixtureDensity md = MixtureDensity.readFromFile(args[i]);
			double [] sv = md.superVector(p, m, c);
			if (fw == null)
				fw = new FrameWriter(sv.length);
			fw.write(sv);
		}
		
		if (fw != null)
			fw.close();
	}
}
