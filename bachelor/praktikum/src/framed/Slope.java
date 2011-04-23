package framed;

import java.io.IOException;

import exceptions.MalformedParameterStringException;

/**
 * Use a Slope object to compute derivatives of a static input frame. Currently
 * supported are derivatives up to the order of 3. Use the Context class to
 * specify the desired derivatives and their context. Each derivative may be
 * computed over a different context. Tirol smoothing (x[i] = x[i-1]/4 + x[i]/2 + x[i+1]/4)
 * is enabled by default.
 * 
 * @author sikoried
 *
 */
public class Slope implements FrameSource {
	
	/** Source to read from */
	private FrameSource source = null;
	
	/** incoming frame size */
	private int fs_in = 0;
	
	/** outgoing frame size */
	private int fs_out = 0;
	
	/** index within ringbuf for next write */
	private int ind_read = 0;
	private int ind_write = 0;
	
	/** longest context to consider; at least one to allow for tirol smoothing */
	private int lc = 3;
	
	/** apply tirol smoothing? */
	private boolean tirol = false;
	
	/** internal ring buffer for context caching */
	private double [][] ringbuf = null;
	
	/**
	 * The Context class holds all information necessary to compute derivatives.
	 */
	public static class Context {
		/**
		 * Generate a new context for derivatives
		 * @param context number of frames considered (total)
		 * @param order order of derivative
		 */
		public Context(int context, int order) {
			if (context < 3 || context % 2 == 0)
				throw new RuntimeException("context is required to be odd and > 2");
			
			if (order > context)
				throw new RuntimeException("context must be larger than order of derivative");
			
			this.context = context;
			this.tau = context/2;
			this.order = order;
			
			cacheRegressionWeights();
		}
		
		/**
		 * Generate a new context for derivatives
		 * @param context number of frames considered (total)
		 * @param order order of derivative
		 * @param scale apply a scaling factor to the derivatives
		 */
		public Context(int context, int order, double scale) {
			if (context < 3 || context % 2 == 0)
				throw new RuntimeException("context is required to be odd and > 2");
			
			if (order > context)
				throw new RuntimeException("context must be larger than order of derivative");
			
			this.scale = scale;
			this.context = context;
			this.tau = context/2;
			this.order = order;
			
			cacheRegressionWeights();
		}
		
		/** scaling factor */
		double scale = 1.;
		
		/** size of the context (in total!) */
		int context;
		
		/** context/2 */ 
		int tau;
		
		/** order of derivative (1...3) */
		int order;
		
		/** cached regression weights */
		double [] rho;
		
		/** cached regression denominator */
		double denom = 0.;
		
		/** cache the regression weights */
		private void cacheRegressionWeights() {
			rho = new double [context];
			denom = 0.;
			for (int j = -tau; j <= tau; ++j) {
				rho[j + tau] = regressionPolynome(order, j, context);
				denom += rho[j + tau] * rho[j + tau];
			}
			denom = 1./denom;
		}
		
		/**
		 * Compute the regression polynomial value for the r-th derivation (see 
		 * Schukat Chap. 3.3 Eq. 3.65
		 * @param r Order of the derivation
		 * @param t context-index
		 * @param c context size (usually 3 or 5)
		 * @return value of the regression polynomial 
		 */
		private double regressionPolynome(int r, int j, int c) {
			double alpha = c; // tau = context/2 + 1
			double t = (double) j;
			if (r == 0)
				return 1;
			else if (r == 1)
				return j;
			else if (r == 2) 
				return t * t - (alpha * alpha - 1.) / 12.; 
			else if (r == 3)
				return t * t * t - (3. * alpha * alpha - 7) * t * .05;
			else
				throw new RuntimeException("unsupported order of derivative");
		}
				
		public String toString() {
			return "[" + context + ":" + order + "]";
		}
	}
	
	/** list of contexts specifying which derivatives to compute */
	private Context [] contexts = { new Context(3, 1) };
		
	/**
	 * Return the outgoing frame size
	 */
	public int getFrameSize() {
		return fs_out;
	}
	
	/**
	 * Enable/disable Tirol filtering (smoothing); (x[i] = x[i-1]/4 + x[i]/2 + x[i+1]/4)
	 * @param tirol
	 */
	public void setTirol(boolean tirol) {
		this.tirol = tirol;
	}
	
	/** 
	 * Tirol smoothing enabled?
	 * @return
	 */
	public boolean getTirol() {
		return tirol;
	}
	
	/**
	 * Generate the default Slope object; computes the first derivative over
	 * a context of 3 frames.
	 * @param source
	 */
	public Slope(FrameSource source) {
		this.source = source;
		fs_in = source.getFrameSize();
		initialize();
	}
	
	/**
	 * Generate a specific Slope object for the given contexts
	 * @see Context
	 * @param source
	 * @param contexts derivatives to compute
	 */
	public Slope(FrameSource source, Context [] contexts) {
		this.source = source;
		fs_in = source.getFrameSize();
		this.contexts = contexts;
		initialize();
	}
	
	/** index where the padding started */
	private int initial_padding = -1;
	
	/**
	 * Read and process the next frame. There are three states: beginning-of-stream, 
	 * in-stream and end-of-stream. The first read call fills the context using
	 * the specified padding method. While in-stream, computation is trivial.
	 * When reaching end-of-stream, the index is remembered, and a new padding 
	 * frame is inserted until no genuine data is available.
	 */
	public boolean read(double[] buf) throws IOException {
		
		// stage1: beginning-of-stream; initialize, read right context and pad left context!
		if (ringbuf == null) {
			ringbuf = new double [lc][fs_in];
			
			// read center frame and right context
			for (int i = 0; i <= lc/2; ++i) {
				if (!source.read(ringbuf[lc/2 + i]))
					return false;
			}
			
			// do the padding
			if (strategy == PAD_COPY) {
				for (int i = 0; i < lc/2; ++i)
					System.arraycopy(ringbuf[lc/2], 0, ringbuf[i], 0, fs_in);
			} else
				throw new RuntimeException("unsupported padding strategy");
			
			// set current read position to the middle
			ind_read = lc/2;
			
			// set current write position to 0 (we assume having read lc samples already)
			ind_write = lc-1;
		} else {
			if (!source.read(ringbuf[ind_write])) {
				// stage3a: first encounter of end-of-stream; remember position for later
				if (initial_padding < 0)
					initial_padding = ind_write;
				// stage3b: no more genuine frames? done!
				else if (ind_write == (initial_padding + lc/2) % lc)
					return false;
				
				// stage3c: alright, pad and go on!
				if (strategy == PAD_COPY) {
					// copy from previous frame
					System.arraycopy(ringbuf[(ind_write + lc - 1) % lc], 0, ringbuf[ind_write], 0, fs_in);
				} else
					throw new RuntimeException("unsupported padding strategy");
			}
			// frame read, we're good to go
		}
		
		// if we come till here, the ring buffer is in consistent shape --
		// compute all the required derivatives!
		
		for (int i = 0; i < contexts.length; ++i) {
			int tau = contexts[i].tau;
			double scale = contexts[i].scale;
			double [] rho = contexts[i].rho;
			double denom = contexts[i].denom;
			
			// for all dimensions...
			for (int k = 0; k < fs_in; ++k) {
				double nom = 0.;
				for (int j = -tau; j <= tau; ++j)
					nom += rho[j + tau] * ringbuf[(ind_read + lc + j) % lc][k];
				
				// save value, remember static features take first fs_in values!
				buf[(i+1) * fs_in + k] = nom * denom * scale;
			}
		}

		// don't forget the statics
		if (tirol) {
			for (int i = 0; i < fs_in; ++i)
				buf[i] = 
					.25 * ringbuf[(ind_read + lc - 1) % lc][i] + 
					.5  * ringbuf[ind_read][i] +
					.25 * ringbuf[(ind_read + lc + 1) % lc][i];
		} else {
			System.arraycopy(ringbuf[ind_read], 0, buf, 0, fs_in);
		}
		
		// increment read and write indices
		ind_read = (ind_read + 1) % lc;
		ind_write = (ind_write + 1) % lc;
		
		return true;
	}
	
	/** pad with the most recently read valid data */
	public static int PAD_COPY = 0;
	
	/** strategy to use at beginning and end of stream */
	private int strategy = PAD_COPY;
	
	/** 
	 * Initialize the slope object by determine the frame and buffer sizes
	 */
	private void initialize() {
		// append the derivatives to the static features
		fs_out = fs_in + contexts.length * fs_in;
		
		// get longest context (required for ring buffer)
		for (Context c : contexts)
			if (c.context > lc)
				lc = c.context;
	}
	
	public String toString() {
		StringBuffer buf = new StringBuffer();
		buf.append("Slope: fs_in=" + fs_in + " fs_out=" + fs_out + " smoothing=" + tirol + " deltas=[");
		for (Context c : contexts)
			buf.append(c);
		buf.append("]");
		return buf.toString();
	}
	
	/**
	 * Create a new Slope object which computes the specified deltas on the 
	 * connected source
	 * 
	 * @param source FrameSource to connect to
	 * @param parameterString "context-size:order[,context-size:order,...]
	 * @return
	 * @throws MalformedParameterStringException
	 */
	public static Slope create(FrameSource source, String parameterString) 
		throws MalformedParameterStringException {
		try {
			String [] h = parameterString.split(",");
			Slope.Context [] contexts = new Slope.Context [h.length];
			for (int i = 0; i < h.length; ++i) {
				String [] c = h[i].split(":");
				if (c.length == 2)
					contexts[i] = new Slope.Context(Integer.parseInt(c[0]), Integer.parseInt(c[1]));
				else if (c.length == 3)
					contexts[i] = new Slope.Context(Integer.parseInt(c[0]), Integer.parseInt(c[1]), Double.parseDouble(c[2]));
				else
					throw new MalformedParameterStringException(h[i] + " is not a valid context");
			}
			return new Slope(source, contexts);
		} catch (Exception e) {
			throw new MalformedParameterStringException(e.toString());
		}
	}
	
	public static void main(String [] args) throws Exception {
		// buffer
		double [][] data = new double [100][1];
				
		// generate 440Hz@16kHz
		sampled.Synthesizer gen = new sampled.SineGenerator(16000, 440.);
		
		for (int i = 0; i < data.length; ++i)
			gen.read(data[i]);
		
		// pack to a frame source
		FrameSource syn = new SimulatedFrameSource(data);
		
		Slope.Context [] c = {new Slope.Context(3, 1), new Slope.Context(3, 2), new Slope.Context(3, 3) };
		
		Slope slope = new Slope(syn, c);
		
		double [] buf = new double [slope.getFrameSize()];
		
		while (slope.read(buf)) {
			for (int i = 0; i < buf.length - 1; ++i)
				System.out.print(buf[i] + " ");
			System.out.println(buf[buf.length-1]);
		}
	}

}
