package sampled;

import java.io.*;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;

import javax.sound.sampled.*;

/**
 * Use this class to capture audio directly from a microphone. The default i
 * s 16kHz at 16bit. Other sampling and bit rates are possible, but capturing is 
 * limited to signed, little endian.
 * 
 * Use the main program to list available mixer names; if no mixer name is 
 * specified on instantiation, the Java AudioSystem default is used.
 * 
 * @author sikoried
 *
 */
public class AudioCapture implements AudioSource {

	AudioFormat	af = null;
	AudioInputStream ais = null;
	
	private TargetDataLine tdl = null;
	
	/** memorize the bit rate */
	private int br = 16;
	
	/** memorize the frame size */
	private int fs = 0;
	
	private static int DEFAULT_SAMPLE_RATE = 16000;
	private static int DEFAULT_BIT_RATE = 16;
	
	/** apply pre-emphasis? */
	private boolean preemphasize = false;
	
	/** value required for first frame of pre-emphasis */
	private double s0 = 0.;
	
	/** pre-emphasis factor */
	private double a = AudioFileReader.DEFAULT_PREEMPHASIS_FACTOR;
	
	/** factor to scale signed values to -1...1 */
	private double scale = 1.;
	
	/** default size for internal buffer */
	private static int DEFAULT_INTERNAL_BUFFER = 128;
	
	/** internal buffer as substitute for external */
	private double [] internalBuffer = null;
	
	/** name of the target mixer or null if default mixer */
	public String mixerName = null;
	
	/** For microphone critical applications: do not fall back to default by default */
	private static final boolean DEFAULT_MIXER_FALLBACK = false;
	
	/** if the desired mixer is not available, fall back to default? */
	private boolean defaultMixerFallBack = DEFAULT_MIXER_FALLBACK;
	
	/** remove DC shift if desired */
	private DCShiftRemover dc = null;
	
	/**
	 * Create an AudioCapture object reading from the specified mixer. To obtain
	 * a mixer name, use AudioCapture.main and call w/ argument "-L"
	 * @param mixerName Name of the mixer to use
	 * @param defaultMixerFallBack fall back to default mixer if desired mixer not available
	 * @throws IOException
	 */
	public AudioCapture(String mixerName, boolean defaultMixerFallBack) 
		throws IOException {
		this.defaultMixerFallBack = defaultMixerFallBack;
		af = new AudioFormat(
				AudioFormat.Encoding.PCM_SIGNED,
				DEFAULT_SAMPLE_RATE, 
				DEFAULT_BIT_RATE, 
				1, 
				DEFAULT_BIT_RATE/8, 
				DEFAULT_SAMPLE_RATE, 
				false
		);
		this.mixerName = mixerName;
		initialize();
	}
	
	/**
	 * Create an AudioCapture object reading from the specified mixer. To obtain
	 * a mixer name, use AudioCapture.main and call w/ argument "-L"
	 * @param mixerName Name of the mixer to use; searches for any occurance of 
	 * 'mixerName' in the available mixer names
	 * @throws IOException
	 */
	public AudioCapture(String mixerName) 
		throws IOException {
		this(mixerName, DEFAULT_MIXER_FALLBACK);
	}
	
	/**
	 * Create an AudioCapture object reading from the specified mixer using 
	 * the given sample rate and bit rate. To obtain a mixer name, use AudioCapture.main
	 * and call w/ argument "-L"
	 * @param mixerName Name of the mixer to read from
	 * @param defaultMixerFallBack fall back to default mixer if desired mixer not available
	 * @param bitRate bit rate to use (usually 8 or 16 bit)
	 * @param sampleRate sample rate to read (usually 8000 or 16000)
	 * @throws IOException
	 */
	public AudioCapture(String mixerName, boolean defaultMixerFallBack, int bitRate, int sampleRate)
		throws IOException {
		this.defaultMixerFallBack = defaultMixerFallBack;
		this.mixerName = mixerName;
		af = new AudioFormat(
				AudioFormat.Encoding.PCM_SIGNED,
				sampleRate, 
				bitRate, 
				1, 
				bitRate/8, 
				sampleRate, 
				false
		);
		initialize();
	}
	
	/**
	 * Create an AudioCapture object reading from the specified mixer using 
	 * the given sample rate and bit rate. To obtain a mixer name, use AudioCapture.main
	 * and call w/ argument "-L"
	 * @param mixerName Name of the mixer to read from
	 * @param bitRate bit rate to use (usually 8 or 16 bit)
	 * @param sampleRate sample rate to read (usually 8000 or 16000)
	 * @throws IOException
	 */
	public AudioCapture(String mixerName, int bitRate, int sampleRate) 
		throws IOException {
		this(mixerName, DEFAULT_MIXER_FALLBACK, bitRate, sampleRate);
	}
	
	/**
	 * Create the default capture object: 16kHz at 16bit
	 * @throws IOException
	 */
	public AudioCapture() 
		throws IOException {
		this(null);
	}
	
	/**
	 * Create a specific capture object (bound to signed, little endian).
	 * @param bitRate target bit rate (usually 8 or 16bit)
	 * @param sampleRate target sample rate (usually 8 or 16kHz)
	 * @throws IOException
	 */
	public AudioCapture(int bitRate, int sampleRate) 
		throws IOException {
		this(null, bitRate, sampleRate);
	}
	
	/**
	 * Initialize the (blocking) capture: query the device, set up and start
	 * the data lines.
	 * @throws IOException
	 */
	private void initialize()
		throws IOException {
		
		br = af.getSampleSizeInBits();
		fs = br/8;
		
		// query a capture device according to the audio format
		DataLine.Info info = new DataLine.Info(TargetDataLine.class, af);
		
		try {
			if (mixerName != null) {
				// query target mixer line
				Mixer.Info [] availableMixers = AudioSystem.getMixerInfo();
				Mixer.Info target = null;
				for (Mixer.Info m : availableMixers)
					if (m.getName().trim().equals(mixerName))
						target = m;
				
				if (target != null)
					tdl = (TargetDataLine) AudioSystem.getMixer(target).getLine(info);
				
				if (tdl == null) {
					if (defaultMixerFallBack) {
						System.err.println("WARNING: couldn't query mixer '" + mixerName + "', falling back to default mixer");
						mixerName = null;
					} else
						throw new IOException("the desired mixer '" + mixerName + "' was not available");
				}
			} 
			
			if (tdl == null) {
				// get default mixer line
				tdl = (TargetDataLine) AudioSystem.getLine(info);
			}
			tdl.open(af);
			tdl.start();
		} catch (LineUnavailableException e) {
			throw new IOException(e.toString());
		}
		
		// set up the audio stream
		ais = new AudioInputStream(tdl);
		
		// compute the scaling factor
		enableScaling();
	}

	/**
	 * Return the current sampling rate
	 */
	public int getSampleRate() {
		return (int)af.getSampleRate();
	}
	
	/**
	 * Tear down the audio capture environment (free resources)
	 */
	public void tearDown() throws IOException {
		tdl.drain();
		tdl.close();
		ais.close();
	}
	
	protected void finalize() throws Throwable {
		try {
			tearDown();
		} finally {
			super.finalize();
		}
	}

	/** the private reading buffer; will be allocated dynamically */
	private byte [] buf = null;

	/**
	 * Read the next buf.length samples (blocking). Samples are normalized to 
	 * [-1;1]
	 * 
	 * @param buf double buffer; will try to read as many samples as fit in the 
	 * buffer. If buf is null, the internal buffer will be used (call enableInternalBuffer
	 * in advance!)
	 * 
	 * @return number of samples actually read
	 * 
	 * @see AudioFileReader.read
	 */
	public int read(double[] buf) 
		throws IOException {
		
		double [] out = (buf == null ? internalBuffer : buf);
			
		
		/* sikoried: I'm not sure how long the OS buffers the data, so the 
		 * processing better be fast ;)
		 */
		
		int ns = out.length;
		
		// memorize buffer size
		if (this.buf == null || this.buf.length != ns*fs)
			this.buf = new byte [ns*fs];
		
		int read = ais.read(this.buf);
		
		// anything read?
		if (read < 1)
			return 0;
		
		// dc shift?
		if (dc != null)
			dc.removeDC(this.buf, read);
		
		// conversion
		if (br == 8) {
			// 8bit: just copy; it's signed and little endian
			for (int i = 0; i < read; ++i) {
				out[i] = scale * (new Byte(this.buf[i]).doubleValue());
				if (out[i] > 1.)
					out[i] = 1.;
				if (buf[i] < -1.)
					out[i] = -1.;
			}			
		} else {
			// > 8bit
			ByteBuffer bb = ByteBuffer.wrap(this.buf);
			bb.order(ByteOrder.LITTLE_ENDIAN);
			int i;
			for (i = 0; i < read / fs; ++i) {
				if (br == 16) {
					out[i] = scale * (double) bb.getShort();
				} else if (br == 32) {
					out[i] = scale * (double) bb.getInt();
				} else
					throw new IOException("unsupported bit rate");
				
				if (out[i] > 1.)
					out[i] = 1.;
				if (out[i] < -1.)
					out[i] = -1.;
			}
			read = i;
		}
		
		if (preemphasize) {
			// set out-dated buffer elements to zero
			if (read < out.length) {
				for (int i = read; i < out.length; ++i)
					buf[i] = 0.;
			}
			
			// remember last signal value
			double help = out[read-1];
			
			AudioFileReader.preEmphasize(out, a, s0);
			
			s0 = help;
		}
		
		return read;
	}
	
	/**
	 * Return a list of Strings matching the mixer names.
	 * @return
	 */
	public static String [] getMixerList() {
		Mixer.Info [] list = AudioSystem.getMixerInfo();
		String [] mixers = new String [list.length];
		for (int i = 0; i < list.length; ++i)
			mixers[i] = list[i].getName();
		return mixers;
	}
	
	/**
	 * Return a string representation of the capture device
	 */
	public String toString() {
		return "AudioCapture: " + (mixerName != null ? "(mixer: " + mixerName + ") " : "") + af.toString();
	}
	
	public boolean getPreEmphasis() {
		return preemphasize;
	}
	
	/**
	 * Enable pre-emphasis with given factor
	 */
	public void setPreEmphasis(boolean applyPreEmphasis, double a) {
		preemphasize = applyPreEmphasis;
		this.a = a;
	}
	
	/**
	 * Return the raw buffer; mind 8/16 bit and signed/unsigned conversion!
	 */
	public byte [] getRawBuffer() {
		return this.buf;
	}
	
	/**
	 * Return the converted buffer containing (normalized) values
	 * @return
	 */
	public double [] getBuffer() {
		return internalBuffer;
	}
	
	/** enables the DC shift or updates the instance for the given context size */
	public void enableDCShift(int contextSize) {
		dc = new DCShiftRemover(this, br, contextSize == 0 ? DCShiftRemover.DEFAULT_CONTEXT_SIZE : contextSize);
	}
	
	/** turn DC shift off */
	public void disableDCShift() {
		dc = null;
	}
	
	/**
	 * Enable the internal Buffer (instead of the local one)
	 * @param bufferSize
	 */
	public void enableInternalBuffer(int bufferSize) {
		if (bufferSize <= 0)
			internalBuffer = new double [DEFAULT_INTERNAL_BUFFER];
		else
			internalBuffer = new double [bufferSize];
	}
	
	/**
	 * Disable the use of the internal buffer (the internal buffer won't be used
	 * if read() receives a valid buffer.
	 */
	public void disableInternalBuffer() {
		internalBuffer = null;
	}
	
	/** 
	 * Disable the [-1;1] scaling to retrieve the original numeric values of the
	 * signal.
	 */
	public void disableScaling() {
		scale = 1.;
	}
	
	/** 
	 * Enable scaling of the signal to [-1;1] depending on its bit rate
	 */
	public void enableScaling() {
		scale = 1. / (2 << (br - 1));
	}
	
	public static final String synopsis = 
		"usage: AudioCapture -r <sample-rate> [options]\n" +
		"Record audio data from an audio device and print it to stdout; supported\n" +
		"sample rates: 16000, 8000\n" +
		"\n" +
		"Other options:\n" +
		"  -L\n" +
		"    List available mixers for audio capture and exit; the mixer name\n" +
		"    or ID can be used to specify the capture device (useful for\n" +
		"    multiple microphones or to enforce a certain device).\n" +
		"  -m <mixder-name>\n" +
		"    Use given mixer for audio input instead of default device\n" +
		"  -a\n" +
		"    Change output mode to ASCII (one sample per line) instead of SSG\n" +
		"  -o <out-file>\n" +
		"    Save output to given file (default: stdout)\n" +
		"  -h\n" +
		"    Display this help text\n";
	
	public static void main(String[] args) throws IOException {
		if (args.length < 1 || args[0].equals("-h")) {
			System.err.println(synopsis);
			System.exit(1);
		}
		
		int sr = 0;
		int br = 16;
		
		boolean ascii = false;
		boolean listMixers = false;
		String outf = null;
		String mixer = null;
		
		// parse args
		for (int i = 0; i < args.length; ++i) {
			if (args[i].equals("-L"))
				listMixers = true;
			else if (args[i].equals("-o"))
				outf = args[++i];
			else if (args[i].equals("-a"))
				ascii = true;
			else if (args[i].equals("-m"))
				mixer = args[++i];
			else if (args[i].equals("-r"))
				sr = Integer.parseInt(args[++i]);
			else if (args[i].equals("-h")) {
				System.err.println(synopsis);
				System.exit(1);
			} else
				System.err.println("ignoring unknown parameter '" + args[i] + "'");
		}
				
		// init the output stream
		OutputStream osw = (outf == null ? System.out : new FileOutputStream(outf));
		
		// only list mixers and exit
		if (listMixers) {
			for (String m : getMixerList())
				osw.write((m + "\n").getBytes());
			osw.flush();
			osw.close();
			System.exit(0);
		}
		
		// there's actually gonna be recording, init source!
		if (!(sr == 16000 || sr == 8000)) {
			System.err.println("unsupported sample rate; choose 16000 or 8000");
			System.exit(1);
		}		
		AudioSource as = new AudioCapture(mixer, br, sr);
		
		System.err.println(as.toString());
		
		// use any reasonable buffer size, e.g. 256 (about 16ms at 16kHz)
		double [] buf = new double [256];
		while (as.read(buf) > 0) {
			if (ascii) {
				for (double d : buf)
					osw.write(("" + d + "\n").getBytes());
			} else {
				// convert double into 2byte sample
				for (double d : buf) {
					short s = new Double(d*((double)Short.MAX_VALUE + 1)).shortValue();
					byte [] b = new byte[] { (byte)(s & 0x00FF), (byte)((s & 0xFF00)>>8) };
					osw.write(b);
				}
			}
			osw.flush();
		}
		
		osw.close();
	}

}
