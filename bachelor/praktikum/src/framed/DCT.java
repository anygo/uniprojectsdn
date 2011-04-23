package framed;

import java.io.IOException;
import edu.emory.mathcs.jtransforms.dct.DoubleDCT_1D;

public class DCT implements FrameSource {
	
	/** the frame source to read from */
	private FrameSource source = null;
	
	/** FFT object */
	private DoubleDCT_1D dct = null;
	
	/** perform scaling? */
	private boolean scale = false;
	
	/** compute the short time energy */
	private boolean cste = true;
	
	/**
	 * Construct a new FFT object. Frame size stays unchanged, first coefficient
	 * is replaced by the short time energy (in case of a Mel filter bank input
	 * the sum over the bands)
	 */
	public DCT(FrameSource source, boolean scale) {
		this.source = source;
		this.scale = scale;
		
		// init DCT
		dct = new DoubleDCT_1D(source.getFrameSize());
	}
	
	public DCT(FrameSource source, boolean scale, boolean computeShortTimeEnergy) {
		this(source, scale);
		this.cste = computeShortTimeEnergy;
	}
	
	public int getFrameSize() {
		return source.getFrameSize();
	}
	
	public String toString() {
		return "dct: frame_size="+ source.getFrameSize() + " scale=" + scale + " short_time_energy=" + cste;
	}
	
	/**
	 * Read the next frame and apply DCT.
	 */
	public boolean read(double[] buf) 
		throws IOException {
		
		// read frame from source
		if (!source.read(buf))
			return false;
		
		// compute short time energy? 
		double ste = 0.;
		if (cste) {
			for (double d : buf)
				ste += d;
		}
		
		// do dct in-place
		dct.forward(buf, scale);

		// replace first coefficient
		if (cste) {
			buf[0] = ste;
		}
		
		return true;
	}
}
