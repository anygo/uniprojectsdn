package framed;

import java.io.IOException;

/**
 * The EnergyDetector object reads from a FramesSources, typically a window, and
 * returns a window if the energy is higher than a certain threshold
 * 
 * @author bocklet
 * 
 */
public class EnergyDetector implements FrameSource {

	private FrameSource source;
	private double threshold;
	private int fs;

	public EnergyDetector(FrameSource source) {
		this.source = source;
	}

	/**
	 * Initializes the EnergyDector
	 * 
	 * @param source
	 *            FrameSource to read from
	 * @param threshold
	 *            threshold for voice activity decision
	 */
	public EnergyDetector(FrameSource source, double threshold) {

		this.source = source;
		fs = source.getFrameSize();
		this.threshold = threshold;
	}

	/**
	 * Return the outgoing frame size
	 */
	public int getFrameSize() {
		return source.getFrameSize();
	}

	/**
	 * Read from the given source as long as a window is found, that is higher
	 * than the specified threshold
	 * 
	 * @param buf
	 *            buffer to save the signal frame
	 * @return true on success, false if the audio stream terminated before a
	 *         window with sufficient energy could be found
	 */
	public boolean read(double[] buf) throws IOException {

		while (source.read(buf)) {
			/* calculate energy for the entire window */
			double energy = 0.0;
			for (int i = 0; i < fs; i++) {
				energy += buf[i] * buf[i];
			}
			energy /= fs;
			if (energy > threshold) {
				return true;
			}
		}
		return false;
	}

	/**
	 * E
	 * than the specified threshold
	 * 
	 * @param buf
	 *            buffer to save the signal frame
	 * @return true on success, false if the audio stream terminated before a
	 *         window with sufficient energy could be found
	 */
	public static double calcThresholdFromSilence(FrameSource source) throws IOException {

		double[] buf = new double[source.getFrameSize()];
		double sum = 0.0;
		int n = 0;
		while (source.read(buf)) {
			double energy = 0.0;
			for (int i = 0; i < source.getFrameSize(); i++) {
				energy += buf[i] * buf[i];
			}
			energy /= source.getFrameSize();
			System.out.println(energy);
			sum += energy;
			n++;
		}
		sum /= n;
		return sum;
	}

}
