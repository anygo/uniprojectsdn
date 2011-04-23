package framed;

import io.FrameReader;
import io.FrameWriter;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.FileReader;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.util.ArrayList;

import util.Pair;

/**
 * Perform a mean and variance normalization to the incoming feature vector.
 * 
 * @author sikoried
 */
public class MVN implements FrameSource {
	/** FrameSource to read from */
	private FrameSource source;
	
	public MVN() {
		// nothing to do
	}
	
	public MVN(FrameSource src) {
		setFrameSource(src);
	}
	
	public MVN(FrameSource src, String parameterFile) throws IOException, ClassNotFoundException {
		setFrameSource(src);
		loadFromFile(parameterFile);
	}
	
	/** number of samples that contributed to the statistics */
	private long samples;
	
	/** mean values to subtract */
	private double [] means;
	
	/** variances */
	private double [] variances;
	
	/** sigmas for normalization (sqrt(var)) */
	private double [] sigmas;

	/**
	 * Return the current frame size
	 */
	public int getFrameSize() {
		return source.getFrameSize();
	}

	/**
	 * Set the FrameSource to read from.
	 * @param src Valid FrameSource instance.
	 */
	public void setFrameSource(FrameSource src) {
		source = src;
	}
	
	/**
	 * Read the next frame from the source, normalize for zero mean and uniform
	 * standard deviation, and output the frame.
	 */
	public boolean read(double[] buf) throws IOException {
		// read, return false if there wasn't any frame to read.
		if (!source.read(buf))
			return false;
		
		// mean and variance normalization
		for (int i = 0; i < buf.length; ++i)
			buf[i] = (buf[i] - means[i]) / sigmas[i];

		return true;
	}
	
	/** 
	 * Reset all internal statistics to clear the normalization parameters.
	 */
	public void resetStatistics() {
		samples = 0;
		means = null;
		variances = null;
		sigmas = null;
	}
	
	/**
	 * Add samples from the given source to the normalization statistics. Initialize
	 * the parameters if necessary.
	 * @param src
	 * @throws IOException
	 */
	public void extendStatistics(FrameSource src) throws IOException {
		int fs = src.getFrameSize();
		double [] nmeans = new double [fs];
		double [] nvariances = new double [fs];
		long nsamples = 0;
		
		// step 1: accumulate new statistics
		double [] buf = new double [fs];
		while (src.read(buf)) {
			nsamples++;
			for (int i = 0; i < fs; ++i) {
				nmeans[i] += buf[i];
				nvariances[i] += (buf[i] * buf[i]);
			}
		}
		
		if (means == null) {
			// step 2a: set the new statistics
			samples = nsamples;
			means = nmeans;
			variances = nvariances;
			
			for (int i = 0; i < fs; ++i) {
				means[i] /= samples;
				variances[i] = variances[i] / samples - means[i] * means[i];
			}
		} else {
			// step 2b: combine old and new statistics
			if (means.length != nmeans.length)
				throw new IOException("frame dimensions do not match: means.length = " + means.length + " src.getFrameSize() = " + src.getFrameSize());
			
			for (int i = 0; i < fs; ++i) {
				// attention: the variance computation involved a mean (square) subtraction!
				variances[i] += means[i] * means[i];
				
				means[i] = (means[i] * samples + nmeans[i]) / (samples + nsamples);
				variances[i] = (variances[i] + nvariances[i]) / (samples + nsamples) - means[i] * means[i];
			}
			
			// don't forget to update the number of samples for these statistics
			samples += nsamples;
		}
		
		// step 3: compute sigmas
		if (sigmas == null)
			sigmas = new double [variances.length];
		
		for (int i = 0; i < variances.length; ++i)
			sigmas[i] = Math.sqrt(variances[i]);
	}
	
	/**
	 * Read the normalization parameters from the referenced file.
	 * @param fileName
	 * @throws IOException
	 * @throws ClassNotFoundException
	 */
	public void loadFromFile(String fileName) throws IOException, ClassNotFoundException {
		ObjectInputStream ois = new ObjectInputStream(new FileInputStream(new File(fileName)));
		
		// read magic
		String magic = (String) ois.readObject();
		if (!magic.equals("CMVNParameters"))
			throw new IOException("Error reading parameter file: Wrong file format! (magic = " + magic + ")");
		
		samples = ois.readLong();
		means = (double []) ois.readObject();
		variances = (double []) ois.readObject();
		
		sigmas = new double [variances.length];
		for (int i = 0; i < variances.length; ++i)
			sigmas[i] = Math.sqrt(variances[i]);
		
		ois.close();
	}
	
	/**
	 * Save the normalization parameters to the referenced file.
	 * @param fileName
	 * @throws IOException
	 */
	public void saveToFile(String fileName) throws IOException {
		ObjectOutputStream oos = new ObjectOutputStream(new FileOutputStream(new File(fileName)));
		
		// write magic
		oos.writeObject("CMVNParameters");
		
		oos.writeLong(samples);
		oos.writeObject(means);
		oos.writeObject(variances);
		
		oos.close();
	}
	
	/**
	 * Generate a String represenation of the normalization parameters
	 */
	public String toString() {
		StringBuffer ret = new StringBuffer();
		
		ret.append("framed.CMVN samples = " + samples + "\n");
		ret.append("  m = [");
		for (double m : means)
			ret.append(" " + m);
		ret.append(" ]\n  v = [");
		for (double v : variances)
			ret.append(" " + v);
		ret.append(" ]\n");
		
		return ret.toString();
	}
	
	public static final String synopsis = 
		"sikoried, 12-4-2009\n" +
		"Compute a mean and variance normalization for each feature file individually.\n" +
		"Optionally, the normalization parameters can be estimated on all referenced\n" +
		"files (cumulative) or loaded from file. See the options for more details.\n" +
		"\n" +
		"usage: framed.MVN [options]\n" +
		"  --io in-file out-file\n" +
		"    Use the given files for in and output. This option may be used multiple\n" +
		"    times.\n" +
		"  --in-out-list list-file\n" +
		"    Use a list containing lines \"<in-file> <out-file>\" for batch processing.\n" +
		"    This option may be used multiple times.\n" +
		"  --in-list list-file directory\n" +
		"    Read all files contained in the list and save the output to the given\n" +
		"    directory. This option may be used multiple times.\n" +
		"\n" +
		"  --cumulative\n" +
		"    Estimate the MVN parameters on ALL files instead of individual MVN.\n" +
		"  --save-parameters file\n" +
		"    Save the CMVN parameters. This can only be used for single files or in\n" +
		"    combination with --cumulative. In case of --online, the parameters after\n" +
		"    are saved after processing all data.\n" +
		"  --load-parameters file\n" +
		"    Use the CMVN parameters from the given file instead of individual or\n" +
		"    cumulative estimates.\n" +
		"  --simulate\n" +
		"    Only compute the normalization parameters but no data normalization!\n" +
		"\n" +
		"  -h | --help\n" +
		"    Display this help text.\n";
	
	public static void main(String[] args) throws Exception, IOException {
		if (args.length < 2) {
			System.err.println(synopsis);
			System.exit(1);
		}
		
		boolean cumulative = false;
		boolean simulate = false;
		
		String parameterOutputFile = null;
		String parameterInputFile = null;
		
		// store all files to be processed in a list
		ArrayList<Pair<String, String>> iolist = new ArrayList<Pair<String, String>>();
		
		// parse the command line arguments
		for (int i = 0; i < args.length; ++i) {
			if (args[i].equals("-h") || args[i].equals("--help")) {
				System.err.println(synopsis);
				System.exit(1);
			} else if (args[i].equals("--simulate"))
				simulate = true;
			else if (args[i].equals("--cumulative"))
				cumulative = true;
			else if (args[i].equals("--load-parameters"))
				parameterInputFile = args[++i];
			else if (args[i].equals("--save-parameters"))
				parameterOutputFile = args[++i];
			else if (args[i].equals("--io")) {
				// add single file pair
				iolist.add(new Pair<String, String>(args[i+1], args[i+2]));
				i += 2;
			} else if (args[i].equals("--in-list")) {
				BufferedReader lr = new BufferedReader(new FileReader(args[++i]));
				
				// validate output directory
				File outDir = new File(args[++i]);
				if (!outDir.canWrite())
					throw new IOException("Cannot write to directory " + outDir.getAbsolutePath());
				
				// read in the list
				String line = null;
				int lineCnt = 1;
				while ((line = lr.readLine()) != null) {
					String [] help = line.split("/");
					
					// check file
					if (!(new File(line)).canRead())
						throw new IOException(args[i-1] + "(" + lineCnt + "): Cannot read input file " + line);
						
					iolist.add(new Pair<String, String>(line, outDir.getAbsolutePath() + "/" + help[help.length-1]));
					lineCnt++;
				}
			} else if (args[i].equals("--in-out-list")) {
				BufferedReader lr = new BufferedReader(new FileReader(args[++i]));
				String line = null;
				int lineCnt = 1;
				while ((line = lr.readLine()) != null) {
					String [] help = line.split("\\s+");
					
					if (help.length != 2)
						throw new IOException(args[i] + "(" + lineCnt + "): invalid line format");
		
					if (!(new File(help[0])).canRead())
						throw new IOException(args[i] + "(" + lineCnt + "): Cannot read input file " + line);
					
					iolist.add(new Pair<String, String>(help[0], help[1]));
					lineCnt++;
				}
			} else {
				throw new Exception("unknown parameter: " + args[i]);
			}
		}
		
		// check some parameters -- not all combinations make sense!
		if (cumulative == false && iolist.size() > 1 && parameterOutputFile != null)
			throw new Exception("cannot save CMVN parameters for more than 1 file (use --cumulative)");
		
		if (cumulative == true && parameterInputFile != null)
			throw new Exception("cumulative and parameterInputFile are exclusive!");
		
		// system summary
		System.out.println("cumulative: " + cumulative);
		System.out.println("simulate  : " + simulate);
		System.out.println("params-in : " + (parameterInputFile == null ? "none" : parameterInputFile));
		System.out.println("params-out: " + (parameterOutputFile == null ? "none" : parameterOutputFile));
		System.out.println("list-size : " + iolist.size());
		
		MVN work = new MVN();
		
		if (parameterInputFile != null)
			work.loadFromFile(parameterInputFile);
		
		if (cumulative) {
			// read all data
			for (Pair<String, String> p : iolist)
				work.extendStatistics(new FrameReader(p.a));
			
			// save the parameter if required
			if (parameterOutputFile != null)
				work.saveToFile(parameterOutputFile);
			
			if (simulate)
				System.exit(0);
		}
		
		for (Pair<String, String> p : iolist) {
			// for individual CMVN, we need to process the data first -- if not read from file
			if (!cumulative && parameterInputFile == null) {
				work.resetStatistics();
				work.extendStatistics(new FrameReader(p.a));
				
				if (parameterOutputFile != null)
					work.saveToFile(parameterOutputFile);
			}
			
			if (simulate)
				continue;
			
			work.setFrameSource(new FrameReader(p.a));
			FrameWriter fw = new FrameWriter(work.getFrameSize(), p.b);
			double [] buf = new double [work.getFrameSize()];
			
			// read and normalize all samples
			while (work.read(buf))
				fw.write(buf);
			
			fw.close();
		}
	}
}
