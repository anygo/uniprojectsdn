package sampled;

import java.util.ArrayList;
import java.io.IOException;
import exceptions.MalformedParameterStringException;

/**
 * Use the SineGenerator to generate (combinations of) sine waves using specific
 * Hz numbers. Default is endless 440Hz. Pay attention to the multiple constructors,
 * there are long and int variants!
 * 
 * @author sikoried
 *
 */
public final class SineGenerator extends Synthesizer {
	private double [] frequencies = { 440. };
	
	/**
	 * Default Sine generator: 440Hz at Synthesizer.DEFAULT_SAMPLE_RATE = 16000Hz
	 */
	public SineGenerator() {
		super();
	}
	
	/**
	 * Generate 440Hz at Synthesizer.DEFAULT_SAMPLE_RATE = 16000Hz
	 * @param duration length in ms
	 */
	public SineGenerator(long duration) {
		super(duration);
	}
	
	/**
	 * Generate 440Hz at given sample rate
	 * @param sampleRate length in Hz
	 */
	public SineGenerator(int sampleRate) {
		super(sampleRate);
	}
	
	/**
	 * Endless Sine generator at 16kHz
	 * @param frequency target frequency in Hz
	 */
	public SineGenerator(double frequency) {
		super();
		this.frequencies = new double [] { frequency };
	}
	
	/**
	 * Endless Sine generator 
	 * @param frequencies array of frequencies to combine
	 */
	public SineGenerator(double [] frequencies) {
		super();
		this.frequencies = frequencies;
	}
	
	/**
	 * Generate 440Hz at specific sample rate
	 * @param sampleRate in Hz
	 * @param duration in ms
	 */
	public SineGenerator(int sampleRate, long duration) {
		super(sampleRate, duration);
	}
	
	/**
	 * Endless Sine generator
	 * @param frequency frequency in Hz
	 */
	public SineGenerator(int sampleRate, double frequency) {
		super(sampleRate);
		this.frequencies = new double [] { frequency };
	}
		
	/**
	 * Endless Sine generator 
	 * @param sampleRate sample rate in Hz
	 * @param frequencies array of frequencies to combine
	 */
	public SineGenerator(int sampleRate, double [] frequencies) {
		super(sampleRate);
		this.frequencies = frequencies;
	}
	
	/**
	 * Specific Sine generator
	 * @param duration time in ms
	 * @param frequency single frequency
	 */
	public SineGenerator(long duration, double frequency) {
		super(duration);
		this.frequencies = new double [] { frequency };
	}
	
	/**
	 * Specific Sine generator
	 * @param duration time in ms
	 * @param frequencies array of frequencies to combine
	 */
	public SineGenerator(long duration, double [] frequencies) {
		super(duration);
		this.frequencies = frequencies;
	}
	
	/**
	 * Specific Sine generator
	 * @param duration time in ms
	 * @param frequencies array of frequencies to combine
	 */
	public SineGenerator(int sampleRate, long duration, double [] frequencies) {
		super(sampleRate, duration);
		this.frequencies = frequencies;
	}
	
	public void setFrequency(double frequency) {
		this.frequencies = new double [] { frequency };
	}
	
	public void setFrequency(double [] frequencies) {
		this.frequencies = frequencies;
	}
	
	public double [] getFrequency() {
		return frequencies;
	}
	
	public String toString() {
		StringBuffer sb = new StringBuffer();
		sb.append("SineGenerator: sample_rate=" + getSampleRate() + " frequency=[");
		for (double d : frequencies)
			sb.append(" " + d);
		sb.append(" ]");
		return sb.toString();
	}
	
	public void tearDown() throws IOException {
		// nothing to do here
	}

	/**
	 * Generate a sine wave according to the super class's current sample number
	 */
	protected void synthesize(double[] buf, int n) {
		// clear
		for (int i = 0; i < n; ++i)
			buf[i] = 0;
		
		int samples = getSamples();
		double sr = (double) getSampleRate();
		
		// overlay the requested frequencies
		for (double freq : frequencies) {
			double c = 2. * Math.PI * freq / sr;
			for (int i = 0; i < n; ++i) {
				buf[i] += Math.sin((samples+i) * c);
			}
		}
		
		// normalize
		for (int i = 0; i < n; ++i)
			buf[i] *= .3;
	}
	
	/**
	 * Generate a new SineGenerator according to the parameter string.
	 * 
	 * @param parameterString "sample-rate,duration,freq1[,freq2,...]
	 * @return
	 * @throws MalformedParameterStringException
	 */
	public static SineGenerator create(String parameterString) 
		throws MalformedParameterStringException {
		try {
			String [] h = parameterString.split(",");
			int i = 0;
			int sampleRate = Integer.parseInt(h[i++]); 
			long duration = Long.parseLong(h[i++]);
			
			ArrayList<Double> freqs = new ArrayList<Double>();
			for (; i < h.length; ++i)
				freqs.add(Double.parseDouble(h[i]));
			
			if (freqs.size() < 1)
				throw new MalformedParameterStringException("no frequency specified");
			
			double [] f = new double [freqs.size()];
			
			for (i = 0; i < f.length; ++i)
				f[i] = freqs.get(i);
			
			return new SineGenerator(sampleRate, duration, f);
		} catch (Exception e) {
			throw new MalformedParameterStringException(e.toString());
		}	
	}
}
