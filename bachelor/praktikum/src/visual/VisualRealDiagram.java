package visual;

import java.awt.Graphics;
import java.awt.Graphics2D;
import java.io.IOException;

import framed.FrameSource;

/**
 * class extending Frame Visualizer to create real diagrams to display real
 * functions
 * 
 * @author Matthias Ring, Markus Goetze
 * 
 */
public class VisualRealDiagram extends FrameVisualizer {
	private static final long serialVersionUID = 1L;

	/**
	 * maximum frequency of audio signal
	 */
	protected double maxXValue;

	/**
	 * original maximum x value needs to be saved since frame source may have
	 * changed
	 */
	private double originalMaxXValue;

	/**
	 * determines whether to logarithmize the data
	 */
	private boolean log = false;

	/**
	 * buffer of data shown at the moment (may differ to original data in case
	 * of zoom)
	 */
	protected double[] copyBuffer;

	/**
	 * size of copy buffer
	 */
	protected int copyBufferSize;

	/**
	 * factor: values per pixel
	 */
	protected double valsPerPix;

	/**
	 * start frequency of range currently displayed
	 */
	protected double currentMinXValue;

	/**
	 * end frequency of range currently displayed
	 */
	protected double currentMaxXValue;

	/**
	 * x axis label
	 */
	private String xAxisLabel;

	/**
	 * Displaying spectrum or autocorrelation diagram
	 */
	private boolean spectrumDiagram;

	/**
	 * Constructor for Buffered audio source
	 * 
	 * @param title
	 *            Title to be shown
	 * @param log
	 *            determines whether to logarithmize
	 * @param frequency
	 *            maximum frequency
	 */
	protected VisualRealDiagram(String title, boolean log, double maxXValue,
			String xAxisLabel, boolean spectrumDiagram) {
		super(title);
		this.setLog(log);
		setxAxisLabel(xAxisLabel);
		setSpectrumDiagram(spectrumDiagram);
		setOriginalMaxXValue(maxXValue);
	}

	/**
	 * Constructor for stream source
	 * 
	 * @param source
	 *            Source
	 * @param title
	 *            Title to be displayed
	 * @param log
	 *            determines whether to logarithmize
	 * @param frequency
	 *            maximum frequency
	 */
	public VisualRealDiagram(FrameSource source, String title, boolean log,
			double maxXValue, String xAxisLabel, boolean spectrumDiagram) {

		super(source, title);

		if (!spectrumDiagram) {
			setMaxXValue((1.0 / maxXValue) * (double) source.getFrameSize()
					* 1000.0);
		} else {
			setMaxXValue(maxXValue / 1000);
		}
		this.setLog(log);
		this.setxAxisLabel(xAxisLabel);
		this.setSpectrumDiagram(spectrumDiagram);
		this.setOriginalMaxXValue(maxXValue);

		adjustToBufferedSource();
	}

	/**
	 * in case that source is a buffered audio source, buffer features
	 * contuniuosly change and are being adjusted
	 */
	protected void adjustToBufferedSource() {
		super.adjustToBufferedSource();
		copyBuffer = new double[frameSize];
		copyBufferSize = frameSize;
		currentMinXValue = 0;

		if (!spectrumDiagram) {
			setMaxXValue((1.0 / originalMaxXValue)
					* (double) frameSource.getFrameSize() * 1000.0);
		} else {
			setMaxXValue(originalMaxXValue / 1000);
		}
		currentMaxXValue = getMaxXValue();
	}

	/**
	 * sets maximum x value
	 * 
	 * @param x
	 *            value
	 */
	public void setMaxXValue(double maxXValue) {
		this.maxXValue = maxXValue;
	}

	/**
	 * 
	 * @return maximum x value
	 */
	public double getMaxXValue() {
		return maxXValue;
	}

	/**
	 * sets range of values on y-axis
	 * 
	 * values have to be symmetric (also after logarithm)
	 * 
	 * @param min
	 * @param max
	 */
	public void setMinMax(double min, double max) {
		this.max = (isLog() ? Math.log(1E-6 + max) : max);
		this.min = (isLog() ? Math.log(1E-6+ min) : min);
	}

	/**
	 * reads data into buffer and then copies into copy buffer. Logarithmizes if
	 * flag is set
	 * 
	 * @param buf
	 *            Buffer to fill
	 */
	@Override
	public boolean read(double[] buf) throws IOException {
		boolean ret = super.read(buf);

		if (!ret) {
			repaint();
			return false;
		}

		if (isLog()) {
			for (int i = 0; i < frameSize; ++i) {
				this.originalBuffer[i] = Math.log(1E-6 + this.originalBuffer[i]);
			}
		}
		if (spectrumDiagram) {
			normalizeBuffer();
		}
		System.arraycopy(this.originalBuffer, 0, this.copyBuffer, 0, frameSize);
		copyBufferSize = frameSize;
		repaint();

		return ret;
	}

	/**
	 * in case of a narrow window, meaning there is more data than pixels,
	 * calculate median between data displayed in one pixel
	 * 
	 * @param w window size
	 */
	private void computeAveragePixelBuffer(int w) {
		for (int i = 0; i < w; i++) {
			pixelBuffer[i] = 0;
			double index = valsPerPix * i;
			int intIndex = (int) Math.floor(index);
			int start = (int) (intIndex - Math.ceil(valsPerPix / 2));
			int end = (int) (intIndex + Math.ceil(valsPerPix / 2));

			if (start < 0)
				start = 0;
			if (end > copyBufferSize - 1)
				end = copyBufferSize - 1;

			for (int j = start; j < end; j++) {
				pixelBuffer[i] += copyBuffer[j];
			}
			pixelBuffer[i] = pixelBuffer[i] / (end - start + 1);
			pixelBuffer[i] = middleOfVals - pixelBuffer[i] * middleOfVals;
		}
	}

	/**
	 * Computes pixels to be drawn based on data buffer
	 * 
	 * @param w
	 *            width of window considering border indent
	 */
	protected void createPixelBuffer(int w) {
		if (w < 10)
			return;

		pixelBuffer = new double[w];
		
//		double nullLine;
//		nullLine = (0 - min) / Math.abs(max - min);
//		nullLine -= 0.5;
//		nullLine *= 2;	

		if (valsPerPix < 1) {
			//computeInterpolatedPixelBuffer(w);
						
			for (int i = 0; i < w; i++) {
				pixelBuffer[i] = copyBuffer[(int) (i * valsPerPix)];
				pixelBuffer[i] = ((double) middleOfVals) - pixelBuffer[i] * ((double) middleOfVals);
			}
		} else { // compute median
			computeAveragePixelBuffer(w);
		}
	}

	/**
	 * computes new data and repaints the image
	 */
	@Override
	public void repaint() {
		int w = getWidth() - 2 * indentBorder;
		
		middleOfVals = getHeight() / 2;
		
		valsPerPix = (copyBufferSize / ((double) w));
		createPixelBuffer(w);

		super.repaint();
	}

	/**
	 * creates the image and draws it
	 */
	@Override
	public void paint(Graphics g) {
		int w = getWidth() - 2 * indentBorder;
		int h = getHeight();

		if (pixelBuffer.length < 10)
			return;

		image = createImage(getWidth(), getHeight());
		Graphics2D bg = (Graphics2D) image.getGraphics();

		drawBackground(bg, getWidth(), h);
		
		double nullLine;
		nullLine = (0 - min) / Math.abs(max - min);
		nullLine -= 0.5;
		nullLine *= 2;	
		nullLine = (int) (getHeight() / 2 - (getHeight()/2) * nullLine);

		// Coordinate system
		drawCoordinateSystem(bg, getWidth(), h, w,
				roundTwoDecimals(currentMaxXValue) + " " + xAxisLabel,
				roundTwoDecimals(currentMinXValue) + " " + xAxisLabel, "",
				getTitle(), (int) nullLine);

		// draw data
		bg.setColor(DRAWING_COLOR);
		for (int i = 1; i < w; i++) {
			bg.drawLine(i - 1 + indentBorder, (int) pixelBuffer[i - 1], i
					+ indentBorder, (int) pixelBuffer[i]);
		}

		// double buffer
		g.drawImage(image, 0, 0, this);
	}

	/**
	 * sets the logarithmize flag. If this is a VisualAutocorrelationDiagram,
	 * values must not be logarithmized
	 * 
	 * @param log
	 */
	protected void setLog(boolean log) {
		if (this instanceof VisualAutocorrelationDiagram) {
			this.log = false;
		} else {
			this.log = log;
		}
	}

	/**
	 * 
	 * @return if logarithm used
	 */
	protected boolean isLog() {
		return log;
	}

	/**
	 * Set the x axis label
	 * 
	 * @param xAxisLabel
	 */
	public void setxAxisLabel(String xAxisLabel) {
		this.xAxisLabel = xAxisLabel;
	}

	/**
	 * 
	 * @return the x axis label
	 */
	public String getxAxisLabel() {
		return xAxisLabel;
	}

	/**
	 * determines if this is a spectrum or autocorrelation diagram
	 * 
	 * @param spectrumDiagram
	 */
	public void setSpectrumDiagram(boolean spectrumDiagram) {
		this.spectrumDiagram = spectrumDiagram;
	}

	/**
	 * 
	 * @return determines if this is a spectrum or autocorrelation diagram
	 */
	public boolean isSpectrumDiagram() {
		return spectrumDiagram;
	}

	/**
	 * sets the maximum x value
	 * 
	 * @param originalMaxXValue
	 */
	public void setOriginalMaxXValue(double originalMaxXValue) {
		this.originalMaxXValue = originalMaxXValue;
	}

	/**
	 * 
	 * @return the maximum x value
	 */
	public double getOriginalMaxXValue() {
		return originalMaxXValue;
	}
}
