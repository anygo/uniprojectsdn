package framed;

import java.util.List;
import java.util.LinkedList;
import java.io.IOException;

/**
 * Use the SimulatedFrameReader to generate a FrameSource with a predefined
 * sequence of frame values. Usefull for debugging and testing.
 * 
 * @author sikoried
 */
public class SimulatedFrameSource implements FrameSource {
	/** frame size */
	private int fs = 0;
	
	/** simulated data */
	private List<double []> data = new LinkedList<double []>();
	
	/**
	 * Generate a artificial sequence of frames
	 * @param data
	 */
	public SimulatedFrameSource(double [][] data) {
		for (double [] d : data)
			this.data.add(d);
		
		fs = data[0].length;
	}
	
	public int getFrameSize() {
		return fs;
	}

	/**
	 * Read the next frame from the data array
	 */
	public boolean read(double[] buf) throws IOException {
		if (data.size() == 0)
			return false;
		
		// copy data, advance pointer
		double [] out = data.remove(0);
		System.arraycopy(out, 0, buf, 0, fs);
		
		return true;
	}
	
	/**
	 * Adds a frame to the simulated frame source.
	 * @param frame the referenced frame will be copied!
	 */
	public void appendFrame(double [] frame) {
		data.add(frame.clone());
	}
}
