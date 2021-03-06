package visual;

import framed.AutoCorrelation;
import framed.Window;

/**
 * Class extending ZoomableRealDiagram to display a correlation function from a buffered audio source. Attributes of the window being used for FFT may be changed here.
 * @author Matthias Ring, Markus Goetze
 *
 */
public class VisualAutocorrelationDiagram extends ZoomableRealDiagram implements
		FrameListener {

	private static final long serialVersionUID = 1L;
	
	/**
	 * Type of window
	 */
	private String winType;
	
	/**
	 * Length of window
	 */
	private int winLen;
	
	/**
	 * Shift length of window
	 */
	private final int SHIFT_LEN = 10;
	
	/**
	 * buffered audio source to read from 
	 */
	private BufferedAudioSource bas; 
	
	/**
	 * start sample in buffered audio source
	 */
	private int startSample;


	/**
	 * @param title to be used
	 * @param log determines whether to logarithmize
	 * @param maxFrequency maximum frequency
	 */
	public VisualAutocorrelationDiagram(String title,
			double frequency) {
		super(title, false, frequency, "ms", false);
	}

	/**
	 * compute new values, possibly after a change to the FFT window
	 * @throws Exception
	 */
	public void update() throws Exception {
		int length = (int) (bas.getSampleRate() * getWinLen() / 1000.);
		
		BufferedAudioSourceReader as = bas.getReader(startSample, length);
		Window w = Window.create(as, getWinType() + "," + winLen + "," + SHIFT_LEN);

		AutoCorrelation auto = new AutoCorrelation(w);
		frameSource = auto;

		adjustToBufferedSource();
		read(new double[frameSize]);
	}
	
	/**
	 *  This method is being called by FrameChanger classes to change the frame to be displayed
	 * @param bas buffered audio source
	 * @param startSample start sample in buffered audio source
	 */
	@Override
	public void show(BufferedAudioSource bas, int startSample) throws Exception {
		this.bas = bas;
		this.startSample = startSample;
		
		update();
	}

	/**
	 * set window type
	 * @param winType
	 */
	public void setWinType(String winType) {
		this.winType = winType;
	}

	/**
	 * returns window type
	 * @return winType
	 */
	public String getWinType() {
		return winType;
	}

	/**
	 * Set window length
	 * @param winLen
	 */
	public void setWinLen(int winLen) {
		this.winLen = winLen;
	}

	/**
	 * returns window length
	 * @return winLen
	 */
	public int getWinLen() {
		return winLen;
	}
	
	/**
	 * Set window type and window length. Convenience function :)
	 * @param winType
	 * @param winLen
	 */
	public void setWindow(String winType, int winLen) {
		setWinLen(winLen);
		setWinType(winType);
	}
}
