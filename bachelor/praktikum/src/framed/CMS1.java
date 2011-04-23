package framed;

import io.FrameReader;
import io.FrameWriter;

import java.io.*;

/**
 * Simple straight-forward (offline) cepstral mean subtraction. Compute the 
 * mean large amount of speech data to get a reliable value. A good strategy
 * is also to compute the mean on an utterance or speaker basis.
 * 
 * @author sikoried
 * @deprecated
 *
 */
public class CMS1 implements FrameSource {

	/** frame size */
	private int fs = 0;
	
	/** source to read from */
	private FrameSource source = null;
	
	/** mean vector */
	private double [] mean = null;
	
	/** internal buffer */
	private double [] buf = null;
	
	/**
	 * Generate a cepstral mean subtraction object using the given source and
	 * mean vector;  no dynamic update
	 * @param source Source to read from
	 * @param mean Mean value @see loadMeanFromFile
	 */
	public CMS1(FrameSource source, double [] mean) {
		this.source = source;
		this.mean = mean;
		fs = source.getFrameSize();
		buf = new double [fs];
	}
		
	/**
	 * Save the current mean vector to the specified file.
	 * @param fileName
	 * @throws IOException
	 */
	public void saveMeanToFile(String fileName) throws IOException {
		FrameWriter fw = new FrameWriter(fs, fileName);
		double [] out = new double [fs];
		System.arraycopy(mean, 0, out, 0, fs);
		fw.write(out);
		fw.close();
	}
	
	/**
	 * Load a mean vector from the specified file.
	 * @param fileName
	 * @param source connector for the newly created CMS
	 * @return Mean vector
	 */
	public static CMS1 loadMeanFromFile(String fileName, FrameSource source) 
		throws IOException {
		FrameReader fr = new FrameReader(fileName);
		
		double [] mean = new double [fr.getFrameSize()];
		
		if (!fr.read(mean))
			throw new IOException("FrameReader.read failed!");
		
		fr.close();
		
		return new CMS1(source, mean);
	}
	
	/**
	 * Compute the mean vector from the given frame source. sikoried: The 
	 * numeric quality might be low due as there is a full summation with a 
	 * single division in the end.
	 * 
	 * @param source FrameSource to read from
	 * @return number of samples read
	 * @throws IOException
	 */
	public static long computeMeanFromSource(FrameSource source, double [] mean) 
		throws IOException {
		
		int fs = source.getFrameSize();
				
		if (mean.length != fs)
			throw new IOException("mean.length != source.getFrameSize()");
		
		double [] buf = new double [fs];
		
		int n = 0;
		
		// read all samples
		while (source.read(buf)) {
			++n;
			for (int i = 0; i < fs; ++i) {
				mean[i] += buf[i];
			}
		}
		
		for (int i = 0; i < fs; ++i)
			mean[i] /= n;
		
		return n;
	}
	
	public int getFrameSize() {
		return fs;
	}
	
	/**
	 * Return a String representation of the object
	 */
	public String toString() {
		String ret = "CMS:  mean=[ ";
		for (double d : mean)
			ret += d + " ";
		return ret += "]";
	}
	
	/**
	 * Read the next frame, subtract the mean if speech
	 */
	public boolean read(double[] buf) throws IOException {
		if (!source.read(this.buf))
			return false;

		for (int i = 0; i < fs; ++i) {
			buf[i] = this.buf[i] - mean[i];
		}

		return true;
	}
}
