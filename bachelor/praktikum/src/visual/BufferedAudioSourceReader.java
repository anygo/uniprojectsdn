package visual;

import java.io.IOException;

import sampled.AudioSource;

/**
 * A reader for BufferedAudioSource that does not change the audioSource's read index
 * and can be used separately, e.g. in threads
 * @author sicawolf
 *
 */
public class BufferedAudioSourceReader implements AudioSource {
	
	/**
	 * current read position in the BufferedAudioSource
	 */
	private int currentReadIndex;
	
	/**
	 * the first index behind the read area
	 */
	private int stopIndex;
	
	/**
	 * the first index that will be read
	 */
	private int startIndex;
	
	/**
	 * the audio source to read from
	 */
	private BufferedAudioSource source;
	
	/**
	 * Creates a new BufferedAudioSourceReader that can read the whole audio source.
	 * @param audioSource the audio signal to read from
	 */
	public BufferedAudioSourceReader(BufferedAudioSource audioSource) {
		currentReadIndex = 0;
		stopIndex = source.getBufferSize();
		startIndex = 0;
		source = audioSource;
	}
	
	/**
	 * Creates a new BufferedAudioSourceReader that reads the specified area of the complete audio source.
	 * If the area index or length is outside the buffer range, this will be corrected internally.
	 * @param audioSource the audio signal to read from
	 * @param startIndex the first index that will be read
	 * @param numberOfSamples the number of samples that the window contains
	 */
	public BufferedAudioSourceReader(BufferedAudioSource audioSource, int startIndex, int numberOfSamples) {
		source = audioSource;
		this.startIndex = startIndex;
		
		this.stopIndex = startIndex + numberOfSamples;
		
		// correct if necessary:
		if (startIndex < 0) {
			this.startIndex = 0;
		} else if (startIndex > source.getBufferSize()) {
			this.startIndex = audioSource.getBufferSize(); //no samples will be read 
		}
		if (stopIndex < 0) {
			stopIndex = 0; // no samples will be read
		} else if (stopIndex > audioSource.getBufferSize()) {
			stopIndex = audioSource.getBufferSize();
		}
		currentReadIndex = this.startIndex;
	}
	
	/**
	 * Resets the current read position to the original one.
	 */
	public void resetReadIndex() {
		currentReadIndex = startIndex;
	}
	
	/**
	 * Sets the current read position to the specified one.
	 * @param position the position at which to start the next read
	 */
	public void setReadIndex(int position) {
		if (position < startIndex) {
			position = startIndex;
		} else if (position > stopIndex) {
			position = stopIndex;
		}
		currentReadIndex = position;
	}

	@Override
	public boolean getPreEmphasis() {
		return source.getPreEmphasis();
	}

	@Override
	public int getSampleRate() {
		return source.getSampleRate();
	}

	@Override
	public int read(double[] buf) throws IOException {
		int read = 0;
		int stop = currentReadIndex + buf.length;
		for (; currentReadIndex < stop; currentReadIndex++) {
			if (currentReadIndex >= stopIndex) {
				break;
			}
			buf[read] = source.get(currentReadIndex);
			read++;
		}
		return read;
	}

	@Override
	public void setPreEmphasis(boolean applyPreEmphasis, double a) {
		source.setPreEmphasis(applyPreEmphasis, a);
	}

	@Override
	public void tearDown() throws IOException {
		source.tearDown();
	}

}