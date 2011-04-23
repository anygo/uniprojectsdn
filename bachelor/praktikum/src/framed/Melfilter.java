package framed;

import java.io.IOException;
import exceptions.MalformedParameterStringException;
import sampled.AudioSource;


/**
 * Triangular mel-filter with log application in the end (if desired). Note that 
 * the bandwidth is fixed(*), thus the number of filters can be controlled using
 * the overlap parameter. By default, log is applied and the calues are clipped. <br/>
 * 
 * (*) as suggested by Florian Hoenig
 * 
 * @author sikoried
 *
 */
public class Melfilter implements FrameSource {
	/** Default lower boundary frequency (Hz) of mel filter bank */
	public static double DEFAULT_LB = 188.;
	
	/** Default upper boundary frequency (Hz) of mel filter bank */
	public static double DEFAULT_UB = 6071.;
	
	/** Default filter width in mel */
	public static double DEFAULT_FW = 226.79982;
	
	/** Default filter overlap */
	public static double DEFAULT_FO = 0.5;
	
	/** Epsilon constant for logarithm */
	public static double EPSILON = 1E-6;
	
	/** frame source used */
	private FrameSource source = null;
	
	/** sample rate of signal */
	private int sr = 16000;
	
	/** source frame size */
	private int fs = 0;
	
	/** size of filter bank */
	private int sfb = 0;
	
	/** lower boundary frequency (Hz), default 188Hz */
	private double lb = DEFAULT_LB;
		
	/** upper boundary frequency (Hz), default 6071Hz */
	private double ub = DEFAULT_UB;
		
	/** filter width (mel) */
	private double fw = DEFAULT_FW;
	
	/** filter overlap of the triangles ]0;1[*/
	private double fo = DEFAULT_FO;
	
	/** apply log after filter? */
	private boolean logMel = true;
	
	/** certain minimum number of filters? (0 = disabled) */
	private int mnf = 0;
	
	/**
	 * Create the default mel filter bank (bandwidth = 226.79982mel)
	 * @param source FrameSource to read from
	 * @param sampleRate sample rate of the original signal in Hz
	 */
	public Melfilter(FrameSource source, int sampleRate) {
		this(source, sampleRate, true, DEFAULT_FW, DEFAULT_LB, DEFAULT_UB, DEFAULT_FO);
	}
	
	/**
	 * Create the default mel filter bank (bandwidth = 226.79982mel) and apply
	 * the logarithm in the end.
	 * @param source FrameSource to read from
	 * @param sampleRate sample rate of the original signal in Hz
	 * @param logMel if true, log is applied to all values in the end
	 */
	public Melfilter(FrameSource source, int sampleRate, boolean logMel) {
		this(source, sampleRate, logMel, DEFAULT_FW, DEFAULT_LB, DEFAULT_UB, DEFAULT_FO);
	}
	
	/**
	 * Create a specific mel filter bank using the following parameters. To be
	 * used from within Melfilter.create
	 * 
	 * @param source FrameSource to read from
	 * @param sampleRate sample rate of the original signal
	 * @param logMel if true, log is applied to all values in the end
	 * @param filterWidthInMel width of the filter bands in mel (default: 226.79982mel)
	 * @param lowerBoundary lower boundary for the filters in Hz (default: 188.0Hz)
	 * @param upperBoundary upper boundary for the filters in Hz (default: 6071.0Hz)
	 * @param overlap amount of overlap between the triangles (default: 0.5; larger value results in more filters!)
	 *
	 */
	private Melfilter(FrameSource source, int sampleRate, boolean logMel, double filterWidthInMel, double lowerBoundary, double upperBoundary, double overlap) {
		this.source = source;
		this.sr = sampleRate;
		this.logMel = logMel;
		fw = filterWidthInMel;
		fo = (1. - overlap);
		lb = lowerBoundary;
		ub = upperBoundary;
		fs = source.getFrameSize();
		
		// make sure filter's not using frequencies out of sight
		if (ub > sr/2)
			ub = sr/2;
		
		// init buffer
		buf = new double [fs];
		
		initializeFilterBank();
	}
	
	/**
	 * Create a specific mel filter bank using the following parameters. To be 
	 * used from within Melfilter.create
	 * 
	 * @param source FrameSource to read from
	 * @param sampleRate sample rate of the original signal
	 * @param logMel if true, log is applied to all values in the end
	 * @param filterWidthInMel width of the filter bands in mel (default: 226.79982mel)
	 * @param lowerBoundary lower boundary for the filters in Hz (default: 188.0Hz)
	 * @param upperBoundary upper boundary for the filters in Hz (default: 6071.0Hz)
	 * @param minNumFilters minimum number of filters (overlap will be adjusted)
	 */
	private Melfilter(FrameSource source, int sampleRate, boolean logMel, double filterWidthInMel, double lowerBoundary, double upperBoundary, int minNumFilters) {
		this.source = source;
		this.sr = sampleRate;
		this.logMel = logMel;
		fw = filterWidthInMel;
		mnf = minNumFilters;
		lb = lowerBoundary;
		ub = upperBoundary;
		fs = source.getFrameSize();
		
		// make sure filter's not using frequencies out of sight
		if (ub > sr/2)
			ub = sr/2;
		
		// init buffer
		buf = new double [fs];
		
		initializeFilterBank();
	}

	/**
	 * Though a certain overlap is requested at initialization, the actual overlap
	 * might be slightly different to exactly match start and end freqs.
	 * @return actual overlap factor
	 */
	public double getActualFilterOverlap() {
		return (1. - fo);
	}
	
	/** 
	 * The frame size is the size of the filter bank!
	 */
	public int getFrameSize() {
		return sfb;
	}

	/** mel frequency to Hz */
	private double fmel2fHz(double fmel) {
	    return (Math.exp( fmel / 1125.) - 1.) * 700.;
	}

	/** Hz to mel frequency */
	double fHz2fmel(double fHz) {
	    return 1125. * Math.log(1. + fHz / 700.);
	}
	
	/** left frequencies of the filter banks */
	private double [] freq_l = null;
	
	/** center frequencies of the filter banks */
	private double [] freq_c = null;
	
	/** right frequencies of the filter banks */
	private double [] freq_r = null;
	
	/** left indices of the triangulars */
	private int [] ind_l = null;
	
	/** center indices of the triangulars */
	private int [] ind_c = null;
	
	/** right indices of the triangulars */
	private int [] ind_r = null;
	
	/**
	 * Initialize the filter bank: determine the center frequencies and their 
	 * index within the frame and allocate all the buffers.
	 */
	private void initializeFilterBank() {
		// start end end of the filter bank
		double lb_mel = fHz2fmel(lb);
		double ub_mel = fHz2fmel(ub);
		
		// delta frequency: which entries of the spectrum represent which freq?
		double df = (double) sr / 2. / (double) fs;
		
		// determine the number of filters
		sfb = (int) Math.ceil((ub_mel - lb_mel - fw) / (fw * fo) + 1);
		
		// minimum number of filters satisfied?
		if (mnf > 0 && sfb < mnf)
			sfb = mnf;
		
		// in case it didn't match, we need to adjust the overlap a bit
		fo = (ub_mel - lb_mel - fw) / (fw * (sfb-1.));
		
		// initialize buffers
		freq_l = new double [sfb];
		freq_c = new double [sfb];
		freq_r = new double [sfb];
		ind_l = new int [sfb];
		ind_c = new int [sfb];
		ind_r = new int [sfb];
		
		// compute triangular frequencies and indices
		for (int i = 0; i < sfb; ++i) {
			freq_l[i] = fmel2fHz(lb_mel + fo*fw*i);
			ind_l[i] = (int) Math.round(freq_l[i] / df);
			
			freq_c[i] = fmel2fHz(lb_mel + fw/2. + fo*fw*i);
			ind_c[i] = (int) Math.round(freq_c[i] / df);
			
			freq_r[i] = fmel2fHz(lb_mel + fw + fo*fw*i);
			ind_r[i] = (int) Math.round(freq_r[i] / df);
			
			// maybe the last filter goes just too far...
			if (ind_r[i] >= fs)
				ind_r[i] = fs-1;
		}
	}
	
	/** buffer to read from source */
	private double [] buf = null;
	
	public boolean read(double[] buf) 
		throws IOException {
		// still frames available?
		if (!source.read(this.buf))
			return false;
		
		double max = -Double.MAX_VALUE;
		
		// apply filter
		for (int i = 0; i < sfb; ++i) {
			// indices...
			int l = ind_l[i];
			int c = ind_c[i];
			int r = ind_r[i];
			
			double accu = 0., w, w_sum = 0.;
			
			// sum over triangle
			int j;
			for (j = l; j <= c; ++j) {
				w = (double) (j - l) / (double)(c - l);
				accu += w * this.buf[j];
				w_sum += w;
			}
			for (; j < r; ++j) {
				w = (double) (r - j - 1) / (double)(r - c - 1);
				accu += w * this.buf[j];
				w_sum += w;
			}
			
			// normalize and copy result
			buf[i] = accu / w_sum;
			
			// update maximum
			max = max < buf[i] ? buf[i] : max;
		}
		
		// normalize bank energies to [eps;1]
		for (int i = 0; i < sfb; ++i) {
			double tmp = buf[i] / max;
			if (tmp < EPSILON)
				buf[i] = EPSILON;
			else
				buf[i] = tmp;
		}
		
		// apply logarithm if requested
		if (logMel) {
			for (int i = 0; i < sfb; ++i)
				buf[i] = Math.log(buf[i]);
		}
				
		return true;
	}
	
	/** 
	 * get a String representation of the filter bank format 3 lines per 
	 * filter (gnuplot ready): <br/>
	 * 
	 * filterNum 0 startFreq <br/>
	 * filterNum 1 centerFreq <br/>
	 * filterNum 0 endFreq <br/>
	 * 
	 * @return String representation of the filter bank
	 */
	public String printFilterBank() {
		StringBuffer sb = new StringBuffer();
		
		sb.append("# filterbank (" + sfb + " filters)\n");
		sb.append("# 3 lines give one filter\n");
		sb.append("# format is: <filter-num> <weight> <frequency>\n");
		
		// we're doing triangles here...
		for (int i = 0; i < sfb; ++i) {
			sb.append(i + " 0 " + freq_l[i] + "\n");
			sb.append(i + " 1 " + freq_c[i] + "\n");
			sb.append(i + " 0 " + freq_r[i] + "\n");
		}
		
		return sb.toString();
	}
	
	public String toString() {
		StringBuffer sb = new StringBuffer();
		sb.append("filterbank: range=" + lb + "-" + ub + "Hz num_filt=" + sfb + " filter-width=" + fw + "mel overlap=" + (1.-fo) + "\n");
		for (int i = 0; i < sfb; ++i)
			sb.append(i + " " +
				ind_l[i] + " (" + freq_l[i] + "Hz) " +
				ind_c[i] + " (" + freq_c[i] + "Hz) " +
				ind_r[i] + " (" + freq_r[i] + "Hz)\n");
		sb.deleteCharAt(sb.length()-1);
		return sb.toString();
	}
	
	/**
	 * Create a Melfilter object using the given parameter string and connect it
	 * to the source. Default parameter String: "188,6071,226.79982,0.5". If you
	 * wish to keep the default value for any of the fields, put a value below 0.
	 * 
	 * @param source
	 * @param sampleRate
	 * @param parameterString "start-hz,end-hz,width-mel,val"; val < 1: min-overlap, val > 1: min-num filters
	 * @return
	 * @throws MalformedParameterStringException
	 */
	public static Melfilter create(FrameSource source, int sampleRate, String parameterString)
		throws MalformedParameterStringException {
		if (parameterString == null)
			return new Melfilter(source, sampleRate);
		else {
			String [] help = parameterString.split(",");
			double start = Double.parseDouble(help[0]);
			double end = Double.parseDouble(help[1]);
			double width = Double.parseDouble(help[2]);
			double val = Double.parseDouble(help[3]);
					
			// check for requested defaults
			if (start < 0.)
				start = DEFAULT_LB;
			if (end < 0.)
				end = DEFAULT_UB;
			if (width < 0.)
				width = DEFAULT_FW;
			if (val < 0.)
				val = DEFAULT_FO;
			
			// create the filter
			if (val < 1.)
				return new Melfilter(source, sampleRate, true, width, start, end, val);
			else
				return new Melfilter(source, sampleRate, true, width, start, end, (int) val);
		}
	}
	
	public static void main(String [] args) throws Exception {
		AudioSource as = new sampled.SineGenerator(44100, 440.); // new AudioCapture();
		System.err.println(as);
		
		Window w = new HammingWindow(as);
		System.err.println(w);
		
		FrameSource spec = new FFT(w);
		System.err.println(spec);
		
		Melfilter mel = Melfilter.create(spec, as.getSampleRate(), "0,22050,600,10");
		System.err.println(mel);
		System.out.println(mel.printFilterBank());
		
//		double [] buf = new double [mel.getFrameSize()];
//		
//		int n = 0;
//		
//		while (mel.read(buf)) {
//			int i = 0;
//			for (; i < buf.length-1; ++i)
//				System.out.print(buf[i] + " ");
//			System.out.println(buf[i]);
//			n++;
//			
//			if (n == 100)
//				break;
//		}
	}
}
