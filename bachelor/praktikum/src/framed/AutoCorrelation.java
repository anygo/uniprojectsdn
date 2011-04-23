package framed;

import java.io.IOException;

/**
 * Compute the autocorrelation coefficients. Make sure to use a proper window!
 * 
 * @author sikoried
 */
public class AutoCorrelation implements FrameSource {

	/** FrameSource to read from */
	private FrameSource source;

	/** internal read buffer */
	private double[] buf;

	/** frame size */
	private int fs;

	/**
	 * Construct an AutoCorrelation object using the given source to read from.
	 * 
	 * @param source
	 */
	public AutoCorrelation(FrameSource source) {
		this.source = source;
		this.fs = source.getFrameSize();
		this.buf = new double[fs];
	}

	public int getFrameSize() {
		return fs;
	}

	/**
	 * Reads from the window and computes the autocorrelation (the lazy way...)
	 */
	public boolean read(double[] buf) throws IOException {
		if (!source.read(this.buf))
			return false;

		for (int j = 0; j < fs; j++) {
			buf[j] = 0.;
			for (int i = 0; i < fs; ++i) {
				buf[j] += this.buf[(i + j) % fs] * this.buf[i];
			}

			buf[j] /= buf[0];
		}

		
		return true;
	}
}
