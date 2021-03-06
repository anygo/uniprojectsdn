package sampled;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;

class DCShiftRemover {
	/** Default context for measuring the DC shift (in ms)*/
	public static final int DEFAULT_CONTEXT_SIZE = 1000;
	
	/** context size in samples */
	private int css;
	
	/** current mean */
	private long mean = 0;
	
	/** history of samples (ring buffer) */
	private long [] hist;
	
	/** current write index for history */
	private int ind = 0;
	
	/** bit rate used */
	private int fs;
	
	/**
	 * Create a DC shift remover for the given AudioSource
	 * @param source
	 */
	public DCShiftRemover(AudioSource source, int bitRate) {
		this(source, DEFAULT_CONTEXT_SIZE, bitRate);
	}
	
	/**
	 * Create a DC shift remover for the given Audiosource and context size
	 * @param source
	 * @param contextSize context to measure DC shift in ms
	 */
	public DCShiftRemover(AudioSource source, int contextSize, int bitRate) {
		fs = bitRate / 2;
		css = source.getSampleRate() / 1000 * contextSize;
		hist = new long [css];
	}
	
	/**
	 * Read from the AudioSource and apply the DC shift. Note that the shift
	 * requires some runtime to function properly
	 */
	public void removeDC(byte [] buf, int read) throws IOException {
		int transferred = 0;
		
		// mind the bit rate
		if (fs == 1) {
			// 8bit: just copy; it's signed and little endian
			for (int i = 0; i < read; ++i) {
				hist[(ind + i) % css] = (long) buf[i];
				transferred++;
			}
		} else {
			// > 8bit
			ByteBuffer bb = ByteBuffer.wrap(buf);
			bb.order(ByteOrder.LITTLE_ENDIAN);
			int i;
			for (i = 0; i < read / fs; ++i) {
				if (fs == 2) {
					hist[(ind + i) % css] = (long) bb.getShort();
				} else if (fs == 4) {
					hist[(ind + i) % css] = (long) bb.getInt();
				} else
					throw new IOException("unsupported bit rate");
				transferred++;
			}
		}
		
		// get mean
		mean = 0;
		for (long d : hist)
			mean += d / css;
		
		// apply the dc shift, retransform to byte array
		for (int i = 0; i < transferred; ++i) {
			if (fs == 1)
				buf[i] = (byte) (hist[ind + i] - mean);
			else {
				ByteBuffer bb = ByteBuffer.allocate(fs);
				bb.order(ByteOrder.LITTLE_ENDIAN);
				
				if (fs == 2)
					bb.putShort((short)hist[ind + i]);
				else if (fs == 4)
					bb.putInt((int)hist[ind + i]);
				
				System.arraycopy(bb.array(), 0, buf, i*fs, fs);
			}
		}
		
		// increment the ring buffer index
		ind = (ind + transferred) % css;
	}
}
