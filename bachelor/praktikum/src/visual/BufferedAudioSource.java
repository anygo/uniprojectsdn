package visual;

import sampled.*;

import java.io.IOException;

/**
 * A Buffer for the complete Audio data.
 *
 */
public class BufferedAudioSource implements AudioSource {
	
	/**
	 * the source file to read from
	 */
	private AudioSource audioSource;
	
	/**
	 * the last index that has not been read yet
	 */
	private int lastRead;
	
	/**
	 * the values
	 */
	private double[] buffer;
	
	/**
	 * the maximum value that occurs in buffer
	 * (will be computed once only)
	 */
	private double maximum;
	
	/**
	 * the minimum value that occurs in buffer
	 * (will be computed only once)
	 */
	private double minimum;
	
	/**
	 * Gets a new Buffer for the AudioSource file that provides all its values.
	 * Note that if the AudioSource is an AudioFileListReader, then the values of all the files will be kept
	 * in the buffer, and getSampleRate() will return the last file's sample rate.
	 * @param source the AudioFileReader or AudioFileListReader whose values will be buffered
	 * @throws IllegalArgumentException - if the AudioSource is an AudioCapture (because streams cannot be
	 * read to end and buffered).
	 */
	public BufferedAudioSource(AudioSource source) {
		if (source instanceof AudioCapture) {
			throw new IllegalArgumentException("The AudioSourceBuffer will not work for streams.");
		}
		audioSource = source;
		lastRead = 0;
		buffer = null;
		maximum = 0;
		minimum = 0;
		fillBuffer();
	}
	
	/**
	 * Reads all the sample values of the audio source into the internal buffer and closes the file reader.
	 */
	private void fillBuffer() {
		int more = 1;
		double [] newValues = new double[65536*2
		                                 ]; // 32768 = 2 ^ 15, ca 2 sec at 16000 kHz
		buffer = new double[0];
		while (more > 0) {
			try {
				more = audioSource.read(newValues);
			} catch (IOException e) {
				// TODO ? show error on console or create dialog window ?
				break;
			}
			if (more > 0) {
				double [] save = buffer;
				buffer = new double[buffer.length + more];
				// copy old values
				for (int i = 0; i < save.length; i++) {
					buffer[i] = save[i];
				}
				// add new values
				for (int i = save.length; i < buffer.length; i++) {
					buffer[i] = newValues[i - save.length];
				}
			}
		}
		try {
			audioSource.tearDown();
		} catch (IOException e) {
			// TODO ? show error on console or create dialog window or do nothing ?
		}
	}
	
	/**
	 * Returns the number of values that are available in the buffer.
	 * @return the size of the buffer, or 0 if it is not yet filled.
	 */
	public int getBufferSize() {
		if (buffer == null) {
			return 0;
		}
		return buffer.length;
	}

	@Override
	public boolean getPreEmphasis() {
		return audioSource.getPreEmphasis();
	}

	@Override
	public int getSampleRate() {
		return audioSource.getSampleRate();
	}

	@Override
	public int read(double[] buf) throws IOException {
		int read = 0;
		if (buffer == null) {
			return 0;
		}
		// all values are available in buffer, simply read on (copy them from there to buf)
		for (; lastRead < buffer.length; lastRead++) {
			if (read >= buf.length) {
				break;
			}
			buf[read] = buffer[lastRead];
			read++;
		}
		return read;
	}
	
	/**
	 * Creates a new BufferedAudioSourceReader that can be used as a new AudioSource, starting read from the beginning.
	 * @return the BufferedAudioSourceReader that does not change this source's read index and can be used
	 * separately (e.g. in threads)
	 */
	public BufferedAudioSourceReader getReader() {
		return new BufferedAudioSourceReader(this);
	}
	
	/**
	 * Creates a new BufferedAudioSourceReader that reads up to numberOfSamples values from the buffer
	 * beginning at the specified startIndex.
	 * @param startIndex where to start reading
	 * @param numberOfSamples the maximum number of values to read (e.g. frame size)
	 * @return the BufferedAudioSourceReader that will provide the frame
	 */
	public BufferedAudioSourceReader getReader(int startIndex, int numberOfSamples) {
		return new BufferedAudioSourceReader(this, startIndex, numberOfSamples);
	}
	
	@Override
	public void setPreEmphasis(boolean applyPreEmphasis, double a) {
		audioSource.setPreEmphasis(applyPreEmphasis, a);
	}

	@Override
	public void tearDown() throws IOException {
		buffer = null; // delete buffer values!
		// audioSource.tearDown() has already been called in fillBuffer() from the constructor
	}
	
	/**
	 * Gets the sample value at the specified index.
	 * @param index the sample index in the sample buffer
	 * @return the sample value at index
	 */
	public double get(int index) {
		return buffer[index];
	}
	
	/**
	 * Gets the maximum value of the buffer. It will be computed only the first time this method is called.
	 * @return the buffer's maximum
	 */
	public double getMax() {
		if (maximum == 0) {
			if (buffer == null || buffer.length < 1) {
				return 0;
			}
			double max = buffer[0];
			for (int i = 1; i < buffer.length; i++) {
				if (buffer[i] > max) {
					max = buffer[i];
				}
			}
			maximum = max;
			return max;
		}
		return maximum;
	}
	
	/**
	 * Gets the minimum value of the buffer. It will be computed only if this method is called for the first time.
	 * @return the buffer's minimum.
	 */
	public double getMin() {
		if (minimum == 0) {
			if (buffer == null || buffer.length < 1) {
				return 0;
			}
			double min = buffer[0];
			for (int i = 1; i < buffer.length; i++) {
				if (buffer[i] < min) {
					min = buffer[i];
				}
			}
			minimum = min;
			return min;
		}
		return minimum;
	}
	
	/**
	 * Gets the corresponding millisecond value for the specified index.
	 * @param index the sample value index
	 * @return the time in milliseconds when the sample occurs
	 */
	public double getBufferIndexAsMilliseconds(int index) {
		return (double)((index) / (double)audioSource.getSampleRate()) * 1000.0;
	}
	
	/**
	 * Gets the corresponding sample index for the specified milliseconds. 
	 * @param milliseconds the milliseconds that should be converted to a sample buffer index
	 * @return the sample value index
	 */
	public int getMillisecondsAsBufferIndex(double milliseconds) {
		double rate = milliseconds / (double) getBufferIndexAsMilliseconds(buffer.length);
		return (int) (rate * (double) buffer.length);
	}
	
	/*
	/**
	 * tries to read as many doubles into buf as possible, starting from index
	 * (not yet tested)
	 * @param buf the buffer to write to
	 * @param startIndex the first index of the values in the file
	 * @return the number of values that have been read
	 * /
	@Override
	public int read(double[] buf, int startIndex) {
		int read = 0;
		if (startIndex < 0 || startIndex >= buffer.length) {
			return read;
		}
		
		int endIndex = startIndex + buf.length;
		if (endIndex >= buffer.length) {
			endIndex = buffer.length;
		}
		for (int i = startIndex; i < endIndex; i++) {
			buf[i - startIndex] = buffer[i];
			read++;
		}
		
		return read;
	} */
	
	/*
	/**
	 * copies the values from file[startIndex] until file[startIndex + howMany - 1] to a new double[]
	 * (not yet tested)
	 * @param startIndex the first index to copy
	 * @param howMany the number of values to be copied
	 * @return the values copied from the buffer
	 * /
	@Override
	public double[] read(int startIndex, int howMany) {
		int endIndex = startIndex + howMany;
		if (buffer == null || startIndex < 0 || howMany < 1 || howMany > buffer.length || endIndex > buffer.length) {
			return null;
		}
		double [] values = new double[howMany];
		for (int i = startIndex; i < endIndex; i++) {
			values[i - startIndex] = buffer[i];
		}
		return values;
	} */
	
	/*
	/**
	 * Gets the buffer that contains the file's values.
	 * @return the buffer that contains all values
	 * (note that if you change them, then the originals are no longer available)
	 * /
	public double[] bufferValues () {
		return buffer;
	}*/

}