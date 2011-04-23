package framed;

import java.io.IOException;

import edu.emory.mathcs.jtransforms.dht.DoubleDHT_1D;

public class DHT implements FrameSource {

	/** the frame source to read from */
	private FrameSource source = null;
	
	/** FFT object */
	private DoubleDHT_1D dht = null;
	

	/**
	 * Construct a new FFT object. Frame size stays unchanged.
	 */
	public DHT(FrameSource source) {
		this.source = source;
		
		// init FFT
		dht = new DoubleDHT_1D(source.getFrameSize());
	}
	
	public int getFrameSize() {
		return source.getFrameSize();
	}
	
	public String toString() {
		return "dht: frame_size=" + source.getFrameSize();
	}
	
	/**
	 * Read the next frame and apply DHT.
	 */
	public boolean read(double[] buf)
		throws IOException {
		
		// read frame from source
		if (!source.read(buf))
			return false;
		
		// do dht in-place
		dht.forward(buf);

		return true;
	}
}
