package sampled;

import java.io.IOException;

/**
 * Any audio source must support the basic operations: read samples, 
 * provide the sample rate (samples per seconds) and should be printable for 
 * debug purposes. It should also support pre-emphasis.
 * 
 * @see AudioFileReader.preEmphasize
 * 
 * @author sikoried
 *
 */
public interface AudioSource {
	/**
	 * Read buf.length samples from the AudioSource.
	 * @param buf Previously allocated buffer to store the read audio samples.
	 * @return Number of actually read audio samples.
	 */
	public int read(double [] buf) throws IOException;
	
	/**
	 * Get the frame rate
	 * @return number of samples per second
	 */
	public int getSampleRate();
	
	/**
	 * Get a string representation of the source
	 */
	public String toString();
	
	/**
	 * Does the AudioSource perform pre-emphasis?
	 */
	public boolean getPreEmphasis();
	
	/**
	 * Toggle the pre-emphasis of the audio signal
	 * @param applyPreEmphasis apply pre-emphasis?
	 * @param a the pre-emphasis factor: x'(n) = x(n) - a*x(n-1)
	 */
	public void setPreEmphasis(boolean applyPreEmphasis, double a);
	
	/**
	 * Tear down the AudioSource (i.e. release file handlers, etc)
	 */
	public void tearDown() throws IOException;
}
