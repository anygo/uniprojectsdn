package framed;

import java.io.IOException;
import sampled.AudioSource;

import edu.emory.mathcs.jtransforms.fft.DoubleFFT_1D;

public class FFT implements FrameSource {

	/** the frame source to read from */
	private FrameSource source = null;
	
	/** minimum coefficients for FFT, padding w/ zeros if required */
	private static int MINIMUM_FFT_COEFFICIENTS = 512;
	
	/** expand frames to next power of 2? */
	private boolean pad = true;
	
	/** input frame size */
	private int fs_in = 0;
	
	/** fft frame size */
	private int fs_fft = MINIMUM_FFT_COEFFICIENTS;
	
	/** output frame size */
	private int fs_out = 0;
	
	/** internal read buffer */
	private double [] buf_read = null;
	
	/** internal fft buffer */
	private double [] buf_fft = null;
	
	/** FFT object */
	private DoubleFFT_1D fft = null;
	
	/**
	 * Construct a new FFT object. Depending on the source frame size, the output
	 * frame size will be 512 or the next power of 2. Frames will be padded with 
	 * zeros. This is done to allow for a better frequency resolution.
	 * @param source FrameSource to read from
	 */
	public FFT(FrameSource source) {
		this.source = source;
		this.pad = true;
		
		initialize();
	}
	
	/**
	 * Construct a new FFT object. 
	 * @param source FrameSource to read from
	 */
	public FFT(FrameSource source, boolean pad) {
		this.source = source;
		this.pad = pad;
		
		initialize();
	}
	
	/**
	 * Initialize the internal buffers and frame sizes
	 */
	private void initialize() {
		// init internal buffers
		fs_in = source.getFrameSize();
		buf_read = new double [fs_in];
		
		fs_fft = fs_in;
		
		if (pad) {
			// pad to the next power of 2, min 512
			int min = 512;
			
			while (min < fs_fft)
				min = min << 1;
			
			fs_fft = min;
		} else {
			// maybe the frame is larger than the default fft frame?
			if (fs_fft < fs_in)
				fs_fft = fs_in;
		}
		
		fft = new DoubleFFT_1D(fs_fft);
		buf_fft = new double [fs_fft];
		fs_out = fs_fft/2 + 1;
	}
	
	public int getFrameSize() {
		return fs_out;
	}
		
	/**
	 * Read the next frame and apply FFT. The output data size is (in/2 + in%2).
	 */
	public boolean read(double[] buf) 
		throws IOException {
		// read frame from source
		if (!source.read(buf_read))
			return false;
		
		// copy data, pad w/ zeros
		System.arraycopy(buf_read, 0, buf_fft, 0, fs_in);
		for (int i = fs_in; i < fs_fft; ++i)
			buf_fft[i] = 0.;
		
		// compute FFT and power spectrum
		fft.realForward(buf_fft);
		
		// refer to the documentation of DoubleFFT_1D.realForward for indexing!
		buf[0] = Math.abs(buf_fft[0]);
		
		for (int i = 1; i < (fs_fft - (fs_fft % 2))/2; ++i)
			buf[i] = Math.sqrt(buf_fft[2*i]*buf_fft[2*i] + buf_fft[2*i+1]*buf_fft[2*i+1]);
		
		if (fs_fft % 2 == 0)
			buf[fs_fft/2] = Math.abs(buf_fft[1]);
		else
			buf[fs_fft/2] = Math.sqrt(buf_fft[fs_fft-1]*buf_fft[fs_fft-1] + buf_fft[1]*buf_fft[1]);
		
		return true;
	}
	
	public String toString() {
		return "fft: fs_in=" + fs_in + " fs_fft=" + fs_fft + " fs_out=" + fs_out;
	}
	
	public static void main(String [] args) throws Exception {
		AudioSource as = new sampled.AudioFileReader(args[0], true);
		System.err.println(as);
		
		Window w = new HammingWindow(as, 25, 10);
		System.err.println(w);
		
		FrameSource spec = new FFT(w);
		System.err.println(spec);
		
		double [] buf = new double [spec.getFrameSize()];
		
		while (spec.read(buf)) {
			int i = 0;
			for (; i < buf.length-1; ++i)
				System.out.print(buf[i] + " ");
			System.out.println(buf[i]);
		}		
	}	
}
