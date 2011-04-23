package framed;

import java.io.IOException;

/**
 * Voiced/Unvoiced (VUV) detection after (Kiessling, 89) using thresholds for
 * zero crossings, squared mean amplitude and maximum amplitude.
 * 
 * @author sikoried
 */
public class VUVDetection implements FrameSource {

	/** Window to read from */
	private Window source;
	
	/** indicator if the last frame read was voiced or not */
	public boolean voiced;
	
	/**
	 * Construct a VUV detector using the given window and thresholds
	 * @param source
	 * @param vuv1 maximum number zero crossings for voiced
	 * @param vuv2 minimum mean squared amplitude for voiced
	 * @param vuv3 absolute amplitude
	 */
	public VUVDetection(Window source) {
		this.source = source;
	}
	
	/**
	 * Read the next frame from the Window and store the VUV decision in the
	 * field voiced.
	 */
	public boolean read(double [] buf) throws IOException {
		if (!source.read(buf))
			return false;
		
		voiced = true;
		
		return true;
	}
	
	public int getFrameSize() {
		return source.getFrameSize();
	}
}
