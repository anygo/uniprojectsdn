package framed;

import sampled.AudioSource;

/**
 * Rectangular window. Boring.
 * @author sikoried
 *
 */
public class RectangularWindow extends Window {
	
	public RectangularWindow(AudioSource source) {
		super(source);
	}
	
	public RectangularWindow(AudioSource source, int windowLength, int shiftLength) {
		super(source, windowLength, shiftLength);
	}

	protected double [] initWeights() {
		double [] w = new double [nsw];
		for (int i = 0; i < nsw; ++i)
			w[i] = 1.;
		return w;
	}

	public String toString() {
		return "RectangularWindow: " + super.toString();
	}
}
