package sampled;

import java.io.IOException;

/**
 * Use the Synthesizer to generate synthetic audio data for test usage.
 * Extending classes need to implement the synthesize method, utilizing the
 * protected member samples indicating the number of samples passed since init.
 * 
 * @author sikoried
 * 
 */
public abstract class Synthesizer implements AudioSource {

	public static int DEFAULT_SAMPLE_RATE = 16000;
	
	/** sample rate of the synthesizer */
	private int sr = DEFAULT_SAMPLE_RATE;
	
	/** number of samples already read */
	private int samples = 0;
	
	/** duration of the synthesis in samples */
	private long duration = 0;
	
	/** get the number of already synthesized samples (for internal use) */
	protected int getSamples() { return samples; }
	
	/** get the duration of the synthesis (in samples, not ms!) */
	protected long getDuration() { return duration; }
	
	/** sleep time: number of ms to wait before the read() call is returned */
	private int sleep = 0;
	
	/** blocking source: if true, the read() method will take about 1/sampleRate seconds */
	private boolean blockingSource = false;
	
	private int msPerSample = 1000 / sr;
	
	public Synthesizer() {
	
	}
	
	/**
	 * Generate a Symthesizer of certain duration
	 * @param duration time in ms
	 */
	Synthesizer(long duration) {
		this.duration = sr / 1000 * duration;
		this.msPerSample = 1000 / sr;
	}
	
	Synthesizer(int sampleRate) {
		this.sr = sampleRate;
		this.msPerSample = 1000 / sr;
	}
	
	/**
	 * Generate a specific Synthesizer
	 * @param sampleRate in Hz
	 * @param duration in ms
	 */
	Synthesizer(int sampleRate, long duration) {
		this.sr = sampleRate;
		this.duration = sr / 1000 * duration;
		this.msPerSample = 1000 / sr;
	}
	
	public boolean isBlockingSource() {
		return blockingSource;
	}
	
	public void setBlocking(boolean blocking) {
		blockingSource = blocking;
	}
	
	public boolean getPreEmphasis() {
		return false;
	}
	
	public void setPreEmphasis(boolean applyPreEmphasis, double a) {
		throw new RuntimeException("method unimplemented");
	}

	public int getSampleRate() {
		return sr;
	}
	
	public void setSleepTime(int sleep) {
		this.sleep = sleep;
	}
	
	public int getSleepTime() {
		return sleep;
	}
	
	private boolean end_of_stream = false;
	
	/**
	 * This function handles the memory i/o and length of the stream (if
	 * applicable). Calls the virtual synthesize method.
	 * 
	 * @see synthesize
	 */
	public int read(double[] buf) throws IOException {
		if (end_of_stream)
			return 0;
		
		int read = buf.length;
		
		// check for end of stream
		if (duration != 0 && samples + buf.length > duration) {
			end_of_stream = true;
			read = (int)(duration - samples);
		}
		
		// remember timestamp
		long ts = 0;
		
		if (blockingSource)
			ts = System.currentTimeMillis();
		
		// synthesize the signal
		synthesize(buf, read);
		
		// increase counter
		samples += read;
		
		// simulate a blocking audio source, good for visualizations
		try {
			long sleep = this.sleep;
			
			// blocking source? compute the remaining sleep time
			if (blockingSource) {
				sleep = msPerSample*buf.length - (System.currentTimeMillis() - ts);
			}
			
			if (sleep > 0) 
				Thread.sleep(sleep); 
		} catch (Exception e) {
			// nothing to do
		}
		
		return read;
	}
	
	/**
	 * Actual synthesizer to be implemented by extending class. If the 
	 * absolute time is required, use the protected variable samples.
	 * 
	 * @see getSamples
	 * @see getDuration
	 * @see getSampleRate
	 * 
	 * @param buf Buffer to save values to
	 * @param n number of samples to generate (0 < n <= buf.length)
	 * @param to time offset in samples from the beginning
	 */
	protected abstract void synthesize(double [] buf, int n);

	/**
	 * String representation of the actual synthesizer
	 */
	public abstract String toString();

	public static void main(String [] args) throws IOException {
		if (args.length < 2) {
			String synopsis = "usage: Synthesizer duration(ms) freq1 [freq2 ...]> ssg\noutput is 16kHz 16bit ssg";
			System.out.println(synopsis);
			System.exit(1);
		}
		
		double [] freqs = new double [args.length-1];
		for (int i = 1; i < args.length; ++i)
			freqs[i-1] = Double.parseDouble(args[i]);
		
		SineGenerator sg = new SineGenerator(Long.parseLong(args[0]), freqs);
		
		System.err.println(sg);
		
		double [] buf = new double [160];
		while (sg.read(buf) > 0) {
			// convert double into 2byte sample
			for (double d : buf) {
				short s = new Double(d*((double)Short.MAX_VALUE + 1)).shortValue();
				byte [] b = new byte[] { (byte)(s & 0x00FF), (byte)((s & 0xFF00)>>8) };
				System.out.write(b);
			}
			System.out.flush();
		}
		
		
	}
}
