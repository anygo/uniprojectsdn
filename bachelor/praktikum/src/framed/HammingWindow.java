package framed;

import sampled.AudioSource;

/**
 * Hamming window: windowed_signal = signal * (0.54 - 0.46*Math.cos(2. * offset * Math.PI / (length - 1)))
 * @author sikoried
 *
 */
public class HammingWindow extends Window {
	
	public HammingWindow(AudioSource source) {
		super(source);
	}
	
	public HammingWindow(AudioSource source, int windowLength, int shiftLength) {
		super(source, windowLength, shiftLength);
	}
	
	protected double [] initWeights() {
		double [] w = new double [nsw];
		for (int i = 0; i < nsw; ++i)
			w[i] = 0.54 - 0.46 * Math.cos(2. * i * Math.PI / (nsw - 1));
		return w;
	}
	
	public String toString() {
		return "HammingWindow: " + super.toString();
	}
}
