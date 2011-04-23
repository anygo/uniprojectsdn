package framed;

import java.io.IOException;
import exceptions.MalformedParameterStringException;
import sampled.*;


public abstract class Window implements FrameSource {
	AudioSource source;
	
	/** frame length in ms */
	private int wl = 16;
	
	/** frame shift in ms */
	private int ws = 10;
	
	/** number of samples in window */
	protected int nsw;
	
	/** number of samples for shift */
	private int nss;
	
	/** weights of the window */
	private double [] w = null;
	
	/**
	 * Create a default Hamming windos (16ms size, 10ms shift)
	 * @param source AudioSource to read from
	 */
	public Window(AudioSource source) {
		this.source = source;
		updateNumerOfSamples();
	}
	
	/**
	 * Create a Hamming window using given frame shift length
	 * @param source AudioSource to read from
	 * @param windowLength Frame length in milli-seconds
	 * @param shiftLength Shift length in milli-seconds
	 */
	public Window(AudioSource source, int windowLength, int shiftLength) {
		this.source = source;
		setWindowSpecs(windowLength, shiftLength);
	}
	
	/**
	 * Get the number of samples within one frame (i.e. the dimension of the feature vector)
	 * @return number of samples within one frame
	 */
	public int getFrameSize() {
		return nsw;
	}
	
	public int getShift() {
		return ws;
	}
	
	private void setWindowSpecs(int windowLength, int shiftLength) {
		wl = windowLength;
		ws = shiftLength;
		
		// can't be longer than frame length
		if (ws > wl)
			ws = wl;
			
		updateNumerOfSamples();
	}
	
	private void updateNumerOfSamples() {
		int sr = source.getSampleRate();
		nsw = (int) (sr * wl / 1000.);
		nss = (int) (sr * ws / 1000.);
		
		// re-allocate ring buffer to correct size, reset current index
		rb = new double [nsw];
		rb_helper = new double [nss];
		cind = -1;
		
		// initialize the weights
		w = initWeights();
	}
	
	public int getNumberOfFramesPerSecond() {
		return (int)(1000. / ws);
	}
	
	/** ring buffer for internal storage of the signal */
	private double [] rb = null;
	
	/** array to cache the newly read data (nss samples) */
	private double [] rb_helper = null;
	
	/** current index in the ring buffer */
	private int cind = -1;
	
	/** number of padded samples */
	private int ps = 0;
	
	/**
	 * Extract the next frame from the audio stream using a window function
	 * @param buf buffer to save the signal frame
	 * @return true on success, false if the audio stream terminated before the window was filled
	 */
	public boolean read(double [] buf) throws IOException {
		// end of stream?
		if (cind == nsw)
			return false;
		
		int n = 0;
		if (cind < 0) {			
			// initialize the buffer, apply window, return
			n = source.read(rb);
			
			// anythig read?
			if (n <= 0)
				return false;
			
			// apply window function to signal
			cind = 0;
			for (int i = 0; i < nsw; ++i)
				buf[i] = rb[i] * w[i];
			
			// done for now
			return true;
		} else if (ps == 0) {
			// default: read from the source...
			n = source.read(rb_helper);
		}

		if (n == nss) {
			// default: enough frames read
			for (int i = 0; i < nss; ++i)
				rb[(cind+i) % nsw] = rb_helper[i];
		} else {			
			// stream comes to an end, take what's there...
			int i;
			for (i = 0; i < n; ++i)
				rb[(cind+i) % nsw] = rb_helper[i];
			
			// ...and pad with zeros until end; increment the padding counter!
			for (; i < nss; ++i, ++ps)
				rb[(cind+i) % nsw] = 0.;
			
			// if there's more padded values as the window is large, we have no genuine signal anymore
			if (ps >= nsw) {
				cind = nsw;
				return false;
			}
		}
		
		// advance ring buffer index
		cind = (cind + nss) % nsw;
		
		// apply window function to signal
		for (int i = 0; i < nsw; ++i)
			buf[i] = rb[(cind + i) % nsw] * w[i];
	
		return true;
	}
	
	public String toString() {
		return "length=" + wl + "ms (" + nsw + " samples) shift=" + ws + "ms (" + nss + " samples)";
	}

	/** 
	 * Actual window function to be implemented by the subclasses.
	 * @return the weights according to the window function
	 */
	protected abstract double [] initWeights();
	
	/**
	 * Generate a new Window object using the parameter string and AudioSource
	 * 
	 * @param source
	 * @param parameterString "hamm|hann|rect,length-ms,shift-ms"
	 * @return
	 * @throws MalformedParameterStringException
	 */
	public static Window create(AudioSource source, String parameterString) 
		throws MalformedParameterStringException {
		if (parameterString == null)
			return new HammingWindow(source);
		else {
			try {
				String [] help = parameterString.split(",");
				int length = Integer.parseInt(help[1]);
				int shift = Integer.parseInt(help[2]);
				if (help[0].equals("hamm"))
					return new HammingWindow(source, length, shift);
				else if (help[0].equals("hann"))
					return new HannWindow(source, length, shift);
				else if (help[0].equals("rect"))
					return new RectangularWindow(source, length, shift);
				else 
					throw new MalformedParameterStringException("unknown window");
			} catch (Exception e) {
				throw new MalformedParameterStringException(e.toString());
			}
		}
	}
	
	public static void main(String [] args) throws Exception {
		AudioSource as = new sampled.AudioFileReader(args[0], RawAudioFormat.create(args.length > 1 ? args[1] : "f:" + args[0]), true); 
		Window window = new HammingWindow(as, 25, 10);
		
		System.err.println(as);
		System.err.println(window);
		
		double [] buf = new double [window.getFrameSize()];

		while (window.read(buf)) {
			int i = 0;
			for (; i < buf.length-1; ++i) 
				System.out.print(buf[i] + " ");
			System.out.println(buf[i]);
		}
	}
}
