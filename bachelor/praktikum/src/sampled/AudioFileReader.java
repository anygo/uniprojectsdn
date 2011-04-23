package sampled;

import java.io.*;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import javax.sound.sampled.*;
import exceptions.MalformedParameterStringException;

/**
 * Use the AudioFileReader to read from an audio file. Supported are all kinds 
 * of raw, Mu- and A-law compressed data in various frame and bit rates. Check
 * the RawAudioFormat class for more details on supported file types. Can use
 * the WAV headers to automatically adjust the parameters.
 * 
 * @author sikoried
 *
 */
public final class AudioFileReader implements AudioSource {
	/** The RawAudioFormat (direct access within the package) */
	RawAudioFormat format = null;
	
	/** InputStream to read from; is either a FileInputStream or BufferedInputStream */
	private InputStream is = System.in;
	
	/** Remember the filename */
	private String fileName = null;
	
	/** apply pre-emphasis? */
	private boolean preemphasize = false;
	
	/** value required for first frame of pre-emphasis */
	private double s0 = 0.;
	
	public static double DEFAULT_PREEMPHASIS_FACTOR = 0.95;
	
	/** pre-emphasis factor */
	private double a = DEFAULT_PREEMPHASIS_FACTOR;
	
	/** use BufferedInputStream? */
	private boolean cacheFile = true;
	
	/** did we close the stream yet? */
	private boolean streamClosed = false;
	
	/** scale factor (dependent on the bit rate) */
	private double scale = 0;
	private double scale_help = 0;
	
	/**
	 * Construct an AudioFileReader using WAV header data
	 * @param fileName
	 * @param cacheFile use buffered reader?
	 * 
	 * @throws UnsupportedAudioFileException
	 * @throws IOException
	 */
	public AudioFileReader(String fileName, boolean cacheFile)
		throws UnsupportedAudioFileException, IOException {
		this.cacheFile = cacheFile;
		format = RawAudioFormat.getRawAudioFormatFromFile(fileName);
		loadFile(fileName);
	}
	
	/**
	 * Construct an AudioFileReader using a custom RawAudioFormat
	 * @param fileName
	 * @param format if null, it will be determined by the header
	 * @param cacheFile use buffered reader?
	 * 
	 * @throws UnsupportedAudioFileException
	 * @throws IOException
	 * 
	 * @see RawAudioFormat.create
	 */
	public AudioFileReader(String fileName, RawAudioFormat format, boolean cacheFile)
		throws IOException {
		try {
		if (format == null)
			format = new RawAudioFormat(AudioSystem.getAudioFileFormat(new File(fileName)).getFormat());
		} catch (UnsupportedAudioFileException e) {
			throw new IOException("AudioFileRedaer(): couldn't determine file type!");
		}
		this.format = format;
		this.cacheFile = cacheFile;
		loadFile(fileName);
	}
	
	/**
	 * Construct an AudioFileReader which reads from an already available byte
	 * array.
	 * @param format RawAudioFormat describing the data
	 * @param data byte array with read audio data
	 * @throws IOException
	 */
	public AudioFileReader(RawAudioFormat format, byte [] data)
		throws IOException {
		this.format = format;
		loadArray(data);
	}

	/**
	 * Load the given filename, eat the header bytes is necessary.
	 * @see RawAudioFormat
	 * @param fileName File to load
	 * @throws IOException
	 */
	private void loadFile(String fileName) 
		throws IOException {
		
		this.fileName = fileName;
		
		if (fileName != null) {
			File inputFile = new File(fileName);
			is = new FileInputStream(inputFile);
		}
		
		// when reading files w/ header, we need to get rid of it!
		if (format.hs > 0) {
			byte [] dispose = new byte [format.hs];
			is.read(dispose);
		}
		
		if (cacheFile) {
			is = new BufferedInputStream(is);
		}
		
		// compute scaling factor
		if (format.alaw || format.ulaw) {
			scale = 1. / Math.pow(2, 15);
			scale_help = Math.pow(2, 15);
		} else {
			scale = 1. / Math.pow(2, format.br-1);
			scale_help = Math.pow(2, format.br-1);
		}
	}
	
	/**
	 * Instead of reading a file, read from the referenced byte array. Despite 
	 * stream initialization, this method is the same as loadFile.
	 * @see loadFile
	 * @param data byte array
	 */
	private void loadArray(byte [] data) 
		throws IOException {
		is = new ByteArrayInputStream(data);
		
		// when reading files w/ header, we need to get rid of it!
		if (format.hs > 0) {
			byte [] dispose = new byte [format.hs];
			is.read(dispose);
		}
		
		// compute scaling factor
		if (format.alaw || format.ulaw) {
			scale = 1. / Math.pow(2, 15);
			scale_help = Math.pow(2, 15);
		} else {
			scale = 1. / Math.pow(2, format.br-1);
			scale_help = Math.pow(2, format.br-1);
		}
	}
	
	public String toString() {
		return (fileName == null ? "STDIN" : fileName) + ": " + format.toString();
	}

	/** the private reading buffer; will be allocated dynamically */
	private byte [] buf = null;

	/**
	 * Read a number of samples from the audio file and save it to the given
	 * buffer. Takes care of signedness and endianess. Samples are 
	 * normalized by the bit rate (and thus [-1;1])
	 * 
	 * @param buf double buffer; will try to read as many samples as fit in the buffer
	 * @return number of samples actually read
	 * @throws IOException
	 */
	public int read(double [] buf) 
		throws IOException {
		if (streamClosed)
			return 0;
		
		int ns = buf.length;
		
		// memorize the buffer size, it's likely that it stays the same
		if (this.buf == null || this.buf.length != ns*format.fs)
			this.buf = new byte [ns*format.fs];
		
		// read requested frames
		int read = is.read(this.buf);
		
		// if nothing was read, close the file!
		if (read < 1) {
			is.close();
			return 0;
		}
		
		// For unsigned I/O check for example: http://darksleep.com/player/JavaAndUnsignedTypes.html
		// For u-law/a-law decompression check for example: http://hazelware.luggle.com/tutorials/mulawcompression.html
		
		// decode the numbers
		if (format.br == 8) {
			// 8bit: raw or compressed?
			if (format.alaw) {
				// a-law decoding?
				for (int i = 0; i < read; ++i) {
					int displacement = this.buf[i] < 0 ? 256 + this.buf[i] : this.buf[i];
					buf[i] = (double) ALAW_DECOMPRESSION[displacement];
				}
			} else if (format.ulaw) {
				// u-law decoding?
				for (int i = 0; i < read; ++i) {
					int displacement = this.buf[i] < 0 ? 256 + this.buf[i] : this.buf[i];
					buf[i] = (double) ULAW_DECOMPRESSION[displacement];
				}
			} else {
				// 8bit raw, just convert...
				if (format.signed) {
					// signed is the "common" java thing
					for (int i = 0; i < read; ++i)
						buf[i] = new Byte(this.buf[i]).doubleValue();
				} else {
					// conversion required
					for (int i = 0; i < read; ++i) 
						buf[i] = (double) (0xFF & ((int) this.buf[i]));
				}
			}
		} else {
			// > 8bit, we need build up the number; check for signedness and endianess!
			if (format.signed) {
				// signed is the "common" java thing
				ByteBuffer bb = ByteBuffer.wrap(this.buf);
				
				// default is big endian
				if (format.littleEndian)
					bb.order(ByteOrder.LITTLE_ENDIAN);
				
				// decode the byte stream
				int i;
				for (i = 0; i < read / format.fs; ++i) {
					if (format.br == 16)
						buf[i] = (double) bb.getShort();
					else if (format.br == 32)
						buf[i] = (double) bb.getInt();
					else
						throw new IOException("unsupported bit rate");
				}
				read = i;
			} else {
				// conversion required for unsigned
				int i;
				for (i = 0; i < read / format.fs; ++i) {
					long val = 0;
					if (format.littleEndian) {
						// MSB last
						for (int j = 0; j < format.fs; ++j) {
							val |=  (long) ( (0xFF & ((int) this.buf[i*format.fs + j])) << j*8); // mind the offset
						}
					} else {
						// MSB first
						for (int j = 0; j < format.fs; ++j) {
							val |=  (long) ( (0xFF & ((int) this.buf[i*format.fs + j])) << (format.fs - j - 1)*8);
						}
					}
					buf[i] = (double) (0xFFFFFFFFL & val);
				}
				read = i;
			}
		}
		
		// normalize to -1...1, ensure proper values
		if (format.signed) {
			for (int i = 0; i < buf.length; ++i) {
				buf[i] *= scale;
				if (buf [i] > 1.)
					buf[i] = 1.;
				if (buf[i] < -1.)
					buf[i] = -1.;
			}
		} else {
			for (int i = 0; i < buf.length; ++i) {
				// unsigned requires the shift first
				buf[i] = scale * (buf[i] - scale_help);
				if (buf[i] > 1.)
					buf[i] = 1.;
				if (buf[i] < -1.)
					buf[i] = -1.;
			}
		}
		
		if (preemphasize) {
			// set out-dated buffer elements to zero
			if (read < buf.length) {
				for (int i = read; i < buf.length; ++i)
					buf[i] = 0.;
			}
			
			// remember last signal value
			double help = buf[read-1];
			
			AudioFileReader.preEmphasize(buf, a, s0);
			
			s0 = help;
		}
		
		// if we couldn't read enough, end of file was reached!
		if (read < buf.length) {
			is.close();
			streamClosed = true;
		}
		
		// return number of read samples
		return read;
	}
	
	public void tearDown() {
		try {
			is.close();
		} catch (IOException e) {
			
		}
	}
	
	/// make sure the input file is closed
	protected void finalize() throws Throwable {
		tearDown();
		super.finalize();
	}
	
	/**
	 * Return the sampling rate of the loaded audio file
	 */
	public int getSampleRate() {
		return format.sr;
	}
	
	public boolean getPreEmphasis() {
		return preemphasize;
	}
	
	public void setPreEmphasis(boolean applyPreEmphasis, double a) {
		preemphasize = applyPreEmphasis;
		this.a = a;
	}
	
	private static short ULAW_DECOMPRESSION [] =	{
	     -32124,-31100,-30076,-29052,-28028,-27004,-25980,-24956,
	     -23932,-22908,-21884,-20860,-19836,-18812,-17788,-16764,
	     -15996,-15484,-14972,-14460,-13948,-13436,-12924,-12412,
	     -11900,-11388,-10876,-10364, -9852, -9340, -8828, -8316,
	      -7932, -7676, -7420, -7164, -6908, -6652, -6396, -6140,
	      -5884, -5628, -5372, -5116, -4860, -4604, -4348, -4092,
	      -3900, -3772, -3644, -3516, -3388, -3260, -3132, -3004,
	      -2876, -2748, -2620, -2492, -2364, -2236, -2108, -1980,
	      -1884, -1820, -1756, -1692, -1628, -1564, -1500, -1436,
	      -1372, -1308, -1244, -1180, -1116, -1052,  -988,  -924,
	       -876,  -844,  -812,  -780,  -748,  -716,  -684,  -652,
	       -620,  -588,  -556,  -524,  -492,  -460,  -428,  -396,
	       -372,  -356,  -340,  -324,  -308,  -292,  -276,  -260,
	       -244,  -228,  -212,  -196,  -180,  -164,  -148,  -132,
	       -120,  -112,  -104,   -96,   -88,   -80,   -72,   -64,
	        -56,   -48,   -40,   -32,   -24,   -16,    -8,     0,
	      32124, 31100, 30076, 29052, 28028, 27004, 25980, 24956,
	      23932, 22908, 21884, 20860, 19836, 18812, 17788, 16764,
	      15996, 15484, 14972, 14460, 13948, 13436, 12924, 12412,
	      11900, 11388, 10876, 10364,  9852,  9340,  8828,  8316,
	       7932,  7676,  7420,  7164,  6908,  6652,  6396,  6140,
	       5884,  5628,  5372,  5116,  4860,  4604,  4348,  4092,
	       3900,  3772,  3644,  3516,  3388,  3260,  3132,  3004,
	       2876,  2748,  2620,  2492,  2364,  2236,  2108,  1980,
	       1884,  1820,  1756,  1692,  1628,  1564,  1500,  1436,
	       1372,  1308,  1244,  1180,  1116,  1052,   988,   924,
	        876,   844,   812,   780,   748,   716,   684,   652,
	        620,   588,   556,   524,   492,   460,   428,   396,
	        372,   356,   340,   324,   308,   292,   276,   260,
	        244,   228,   212,   196,   180,   164,   148,   132,
	        120,   112,   104,    96,    88,    80,    72,    64,
	         56,    48,    40,    32,    24,    16,     8,     0
	};
	
	private static short ALAW_DECOMPRESSION [] = {
	     -5504, -5248, -6016, -5760, -4480, -4224, -4992, -4736,
	     -7552, -7296, -8064, -7808, -6528, -6272, -7040, -6784,
	     -2752, -2624, -3008, -2880, -2240, -2112, -2496, -2368,
	     -3776, -3648, -4032, -3904, -3264, -3136, -3520, -3392,
	     -22016,-20992,-24064,-23040,-17920,-16896,-19968,-18944,
	     -30208,-29184,-32256,-31232,-26112,-25088,-28160,-27136,
	     -11008,-10496,-12032,-11520,-8960, -8448, -9984, -9472,
	     -15104,-14592,-16128,-15616,-13056,-12544,-14080,-13568,
	     -344,  -328,  -376,  -360,  -280,  -264,  -312,  -296,
	     -472,  -456,  -504,  -488,  -408,  -392,  -440,  -424,
	     -88,   -72,   -120,  -104,  -24,   -8,    -56,   -40,
	     -216,  -200,  -248,  -232,  -152,  -136,  -184,  -168,
	     -1376, -1312, -1504, -1440, -1120, -1056, -1248, -1184,
	     -1888, -1824, -2016, -1952, -1632, -1568, -1760, -1696,
	     -688,  -656,  -752,  -720,  -560,  -528,  -624,  -592,
	     -944,  -912,  -1008, -976,  -816,  -784,  -880,  -848,
	      5504,  5248,  6016,  5760,  4480,  4224,  4992,  4736,
	      7552,  7296,  8064,  7808,  6528,  6272,  7040,  6784,
	      2752,  2624,  3008,  2880,  2240,  2112,  2496,  2368,
	      3776,  3648,  4032,  3904,  3264,  3136,  3520,  3392,
	      22016, 20992, 24064, 23040, 17920, 16896, 19968, 18944,
	      30208, 29184, 32256, 31232, 26112, 25088, 28160, 27136,
	      11008, 10496, 12032, 11520, 8960,  8448,  9984,  9472,
	      15104, 14592, 16128, 15616, 13056, 12544, 14080, 13568,
	      344,   328,   376,   360,   280,   264,   312,   296,
	      472,   456,   504,   488,   408,   392,   440,   424,
	      88,    72,   120,   104,    24,     8,    56,    40,
	      216,   200,   248,   232,   152,   136,   184,   168,
	      1376,  1312,  1504,  1440,  1120,  1056,  1248,  1184,
	      1888,  1824,  2016,  1952,  1632,  1568,  1760,  1696,
	      688,   656,   752,   720,   560,   528,   624,   592,
	      944,   912,  1008,   976,   816,   784,   880,   848
	};
	
	/**
	 * Perform a pre-emphasis on the given signal vector: x'(n) = x(n) - a * x(n-1)
	 * with s0 = x(-1)
	 * 
	 * @param buf audio frame; in-place transformation!
	 * @param a pre-emphasis factor 
	 * @param s0 value to use for first element
	 */
	public static void preEmphasize(double [] buf, double a, double s0) {
		// in-place transformation, so begin at the end
		for (int i = buf.length-1; i < 0; --i)
			buf[i] -= a * buf[i-1];
		// first element
		buf[0] -= a * s0;
	}
	
	public static String synopsis = 
		"usage: sampled.AudioFileReader [file-with-header | formatString raw-file]\n";
	
	public static void main(String [] args) 
		throws IOException, UnsupportedAudioFileException, MalformedParameterStringException {
		
		if (args.length < 1 || args.length > 2) {
			System.err.println(synopsis);
			System.exit(1);
		}
		
		String format = null;
		String file = args[args.length-1];
		
		if (args.length == 2)
			format = args[0];


		AudioFileReader afr = null;
		
		if (format == null)
			afr = new AudioFileReader(file, true);
		else
			afr = new AudioFileReader(file, RawAudioFormat.create(format), true);
		
		System.err.println(afr);
		
		double [] buf = new double [1];
		
		while (afr.read(buf) > 0) {
			System.out.println(buf[0]);
		}
	}
}
