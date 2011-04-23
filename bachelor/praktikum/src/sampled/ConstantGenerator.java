package sampled;

import java.io.IOException;

public class ConstantGenerator extends Synthesizer {

	private double constant = .3;
	
	public ConstantGenerator() {
		super();
	}
	
	public ConstantGenerator(int duration) {
		super(duration);
	}
	
	public ConstantGenerator(double constant) {
		super();
		this.constant = constant;
	}
	
	public ConstantGenerator(int duration, double constant) {
		super(duration);
		this.constant = constant;
	}
	
	public void setConstant(double constant) {
		this.constant = constant;
	}
	
	public double getConstant() {
		return constant;
	}
	
	protected void synthesize(double[] buf, int n) {
		for (int i = 0; i < n; ++i)
			buf[i] = constant;
	}
	public void tearDown() throws IOException {
		// nothing to do here
	}

	public String toString() {
		return "ConstantGenerator: sample_rate=" + getSampleRate() + " constant=" + constant;
	}

}
