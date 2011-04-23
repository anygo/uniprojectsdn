package framed;

import java.io.IOException;

/**
 * Once we leave the signal (i.e. sampled) level, we deal with frames. The 
 * Implementation of this interface allows for flexible combinations of the 
 * target algorithms. Usually, implementing objects receive an initialized 
 * source to read from.
 * 
 * @author sikoried
 *
 */
public interface FrameSource {
	/**
	 * Extract the next frame from the the source stream using a window function
	 * @param buf buffer to save the frame; implementing objects may depend
	 * on a constant dimensionduring subsequent calls
	 * @return true on success, false if the stream terminated before the window was filled
	 */
	public boolean read(double [] buf) throws IOException;
	
	/**
	 * Return the length of the frames (needed for the read call)
	 */
	public int getFrameSize();
	
	/**
	 * Return a String representation of the FrameSource
	 */
	public String toString();
}
