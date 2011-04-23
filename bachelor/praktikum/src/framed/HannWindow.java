package framed;

import sampled.AudioSource;

/**
 * Hanning window: windowed_signal = signal * (0.5 - 0.5*Math.cos(2. * offset * Math.PI / (length - 1)))
 * @author sikoried
 *
 */
public class HannWindow extends Window {
	
	public HannWindow(AudioSource source) {
		super(source);
	}
	
	public HannWindow(AudioSource source, int windowLength, int shiftLength) {
		super(source, windowLength, shiftLength);
	}

	protected double [] initWeights() {
		double [] w = new double [nsw];
		for (int i = 0; i < nsw; ++i)
			w[i] = 0.5 - 0.5 * Math.cos(2. * i * Math.PI / (nsw - 1));
		return w;
	}

	public String toString() {
		return "HanningWindow: " + super.toString();
	}
}
